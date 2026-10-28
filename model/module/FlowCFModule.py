import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers

def observe_probability_state(x_t: torch.Tensor, mode: str='bernoulli', training: bool=False) -> torch.Tensor:
    probs = x_t.clamp(0.0, 1.0)
    if training or mode == 'bernoulli':
        return torch.bernoulli(probs)
    if mode == 'expectation':
        return probs
    if mode == 'threshold':
        return (probs >= 0.5).float()
    raise ValueError(f'Unsupported observation_mode: {mode}')

def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def timestep_embedding_pi(timesteps: torch.Tensor, dim: int, max_period: int=10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half) * 2 * math.pi
    args = timesteps.float().reshape(-1, 1) * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class TransformerFlowModel(nn.Module):
    """Relation-ranked intermediate-state encoding and graph-propagated reconstruction."""

    def __init__(self, n_items: int, time_emb_size: int, item_popularity_norm: torch.Tensor, item_relation_graph: torch.Tensor=None, d_model: int=64, nhead: int=4, dim_feedforward_multiplier: int=2, num_layers: int=2, dropout: float=0.1, max_seq_len: int=50, use_causal_mask: bool=True, sort_order: str='hot_cold', use_pos_emb: bool=False, use_relation_value_emb: bool=False, relation_film_type: str='none', decoder_type: str='mlp', context_encoder_type: str='transformer', decoder_hidden_dims=None, decoder_use_relation_scores: bool=False, decoder_relation_propagation: str='none', decoder_relation_propagation_weight: float=1.0, aux_target_type: str='item', aux_relation_conv_weight: float=0.0, observation_mode: str='bernoulli'):
        super().__init__()
        self.time_emb_dim = time_emb_size
        self.observation_mode = observation_mode
        self.max_seq_len = max_seq_len
        self.use_causal_mask = use_causal_mask
        self.sort_order = sort_order
        self.use_pos_emb = use_pos_emb
        self.use_relation_value_emb = use_relation_value_emb
        self.relation_film_type = relation_film_type
        self.decoder_type = decoder_type
        if context_encoder_type not in {'transformer', 'mlp'}:
            raise ValueError(f'Unsupported context_encoder_type: {context_encoder_type}')
        self.context_encoder_type = context_encoder_type
        self.decoder_use_relation_scores = decoder_use_relation_scores
        self.decoder_relation_propagation = decoder_relation_propagation
        self.decoder_relation_propagation_weight = decoder_relation_propagation_weight
        self.aux_target_type = aux_target_type
        self.aux_relation_conv_weight = aux_relation_conv_weight
        self.register_buffer('item_pop', item_popularity_norm)
        self.register_buffer('item_relation_graph', item_relation_graph)
        if decoder_relation_propagation not in {'none', 'concat', 'add', 'only'}:
            raise ValueError(f'Unsupported decoder_relation_propagation: {decoder_relation_propagation}')
        if aux_target_type not in {'item', 'relation_conv'}:
            raise ValueError(f'Unsupported aux_target_type: {aux_target_type}')
        self.item_emb = nn.Embedding(n_items, d_model)
        self.pos_emb = nn.Embedding(500, d_model)
        self.relation_value_proj = nn.Linear(1, d_model) if use_relation_value_emb else None
        if relation_film_type == 'relation':
            self.relation_film = nn.Linear(1, d_model * 2)
        elif relation_film_type == 'time':
            self.relation_film = nn.Linear(1, d_model * 2)
        elif relation_film_type == 'relation_time':
            self.relation_film = nn.Linear(2, d_model * 2)
        elif relation_film_type == 'none':
            self.relation_film = None
        else:
            raise ValueError(f'Unsupported relation_film_type: {relation_film_type}')
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * dim_feedforward_multiplier, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) if context_encoder_type == 'transformer' else None
        if context_encoder_type == 'mlp':
            self.context_token_mixer = nn.Parameter(torch.eye(max_seq_len))
            self.register_buffer('context_causal_mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))
            self.context_mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_model, d_model))
        else:
            self.context_token_mixer = None
            self.context_mlp = None
        decoder_item_channels = 1
        if decoder_use_relation_scores:
            decoder_item_channels += 1
        if decoder_relation_propagation == 'concat':
            decoder_item_channels += 1
        decoder_hidden_dims = list(decoder_hidden_dims) if decoder_hidden_dims is not None else [300]
        self.decoder = MLPLayers(layers=[n_items * decoder_item_channels + d_model + time_emb_size] + decoder_hidden_dims + [n_items], dropout=dropout, activation='tanh', last_activation=False)
        if decoder_type in {'itemwise', 'hybrid'}:
            self.itemwise_context = MLPLayers(layers=[d_model + time_emb_size, d_model, d_model], dropout=dropout, activation='tanh', last_activation=False)
            self.itemwise_bias = nn.Parameter(torch.zeros(n_items))
        else:
            self.itemwise_context = None
            self.itemwise_bias = None
        self.aux_decoder = nn.Linear(d_model, d_model)
        self.apply(xavier_normal_initialization)
        if self.relation_film is not None:
            nn.init.zeros_(self.relation_film.weight)
            nn.init.zeros_(self.relation_film.bias)

    @staticmethod
    def _row_minmax_normalize(values: torch.Tensor) -> torch.Tensor:
        values = values.clamp(min=0.0)
        if values.size(1) > 0:
            values = values.clone()
            values[:, 0] = 0.0
        row_min = values.min(dim=1, keepdim=True).values
        row_max = values.max(dim=1, keepdim=True).values
        return (values - row_min) / (row_max - row_min + 1e-08)

    def _propagate_decoder_state(self, c_t: torch.Tensor) -> torch.Tensor:
        if self.item_relation_graph is None:
            return torch.zeros_like(c_t)
        graph = self.item_relation_graph.to(c_t.device)
        if graph.is_sparse:
            propagated = torch.sparse.mm(graph, c_t.transpose(0, 1)).transpose(0, 1)
        else:
            propagated = torch.matmul(c_t, graph.transpose(0, 1))
        return self._row_minmax_normalize(propagated)

    def _relation_convolved_item_embeddings(self) -> torch.Tensor:
        if self.item_relation_graph is None:
            return self.item_emb.weight
        graph = self.item_relation_graph.to(self.item_emb.weight.device)
        if graph.is_sparse:
            rel_emb = torch.sparse.mm(graph, self.item_emb.weight)
        else:
            rel_emb = torch.matmul(graph, self.item_emb.weight)
        rel_emb = F.normalize(rel_emb, dim=-1)
        item_emb = F.normalize(self.item_emb.weight, dim=-1)
        beta = self.aux_relation_conv_weight
        return (1.0 - beta) * item_emb + beta * rel_emb

    def _select_active_items(self, c_t: torch.Tensor, relation_scores: torch.Tensor=None):
        max_active = max(int(c_t.sum(dim=1).max().item()), 1)
        topk_vals, topk_idx = c_t.topk(max_active, dim=-1)
        pad_mask = topk_vals == 0
        seq_len = min(max_active, self.max_seq_len)
        if self.sort_order == 'unordered':
            sorted_items = topk_idx[:, :seq_len]
            pad_mask = pad_mask[:, :seq_len]
        elif self.sort_order in {'relation_desc', 'relation_asc'}:
            if relation_scores is None:
                raise ValueError(f"relation_scores is required when sort_order='{self.sort_order}'")
            sort_scores = relation_scores.gather(1, topk_idx).clone()
            descending = self.sort_order == 'relation_desc'
            sort_scores[pad_mask] = float('-inf') if descending else float('inf')
            order = sort_scores.argsort(dim=-1, descending=descending)
            sorted_items = topk_idx.gather(1, order)[:, :seq_len]
            pad_mask = pad_mask.gather(1, order)[:, :seq_len]
        else:
            pop_scores = self.item_pop[topk_idx].clone()
            if self.sort_order == 'cold_hot':
                pop_scores[pad_mask] = float('+inf')
                order = pop_scores.argsort(dim=-1, descending=False)
            else:
                pop_scores[pad_mask] = float('-inf')
                order = pop_scores.argsort(dim=-1, descending=True)
            sorted_items = topk_idx.gather(1, order)
            pad_mask = pad_mask.gather(1, order)
            num_real = (~pad_mask).sum(dim=1)
            offset = (num_real - seq_len).clamp(min=0)
            src_idx = torch.arange(seq_len, device=c_t.device).unsqueeze(0) + offset.unsqueeze(1)
            sorted_items = sorted_items.gather(1, src_idx)
            pad_mask = pad_mask.gather(1, src_idx)
        num_real = (~pad_mask).sum(dim=1)
        return (sorted_items, pad_mask, num_real)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, relation_scores: torch.Tensor=None) -> torch.Tensor:
        c_t = observe_probability_state(x_t, self.observation_mode, self.training)
        B, N = c_t.shape
        sorted_items, pad_mask, num_real = self._select_active_items(c_t, relation_scores=relation_scores)
        seq_len = sorted_items.size(1)
        max_active = max(int(c_t.sum(dim=1).max().item()), 1)
        selected_emb = self.item_emb(sorted_items)
        if self.use_pos_emb:
            pos_ids = torch.arange(seq_len, device=x_t.device).unsqueeze(0).expand(B, -1)
            selected_emb = selected_emb + self.pos_emb(pos_ids)
        if self.use_relation_value_emb and relation_scores is not None:
            selected_relation = relation_scores.gather(1, sorted_items).unsqueeze(-1)
            selected_relation = relation_scores.gather(1, sorted_items).unsqueeze(-1)
            selected_emb = selected_emb + self.relation_value_proj(selected_relation)
        if self.relation_film is not None and (relation_scores is not None or self.relation_film_type == 'time'):
            selected_relation = relation_scores.gather(1, sorted_items).unsqueeze(-1) if relation_scores is not None else None
            selected_relation = relation_scores.gather(1, sorted_items).unsqueeze(-1)
            if self.relation_film_type == 'time':
                t_token = t.reshape(B, 1, 1).expand(-1, seq_len, -1)
                film_input = t_token
            elif self.relation_film_type == 'relation_time':
                t_token = t.reshape(B, 1, 1).expand(-1, seq_len, -1)
                film_input = torch.cat([selected_relation, t_token], dim=-1)
            else:
                film_input = selected_relation
            film_scale, film_shift = self.relation_film(film_input).chunk(2, dim=-1)
            selected_emb = selected_emb * (1.0 + torch.tanh(film_scale)) + film_shift
        all_pad_rows = pad_mask.all(dim=1)
        if all_pad_rows.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_pad_rows, 0] = False
            selected_emb = selected_emb.clone()
            selected_emb[all_pad_rows, 0] = 0.0
        if self.context_encoder_type == 'mlp':
            masked_emb = selected_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            padded_emb = F.pad(masked_emb, (0, 0, 0, self.max_seq_len - seq_len))
            mixed = F.linear(padded_emb.transpose(1, 2), self.context_token_mixer * self.context_causal_mask).transpose(1, 2)
            context = self.context_mlp(mixed[:, :seq_len])
        else:
            attn_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x_t.device), diagonal=1) if self.use_causal_mask else None
            context = self.transformer(selected_emb, mask=attn_mask, src_key_padding_mask=pad_mask)
        num_active = num_real.clamp(min=1)
        if self.sort_order == 'unordered':
            non_pad = (~pad_mask).unsqueeze(-1).float()
            user_repr = (context * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)
        else:
            last_idx = (num_active - 1).clamp(max=seq_len - 1)
            user_repr = context[torch.arange(B, device=x_t.device), last_idx, :]
        if all_pad_rows.any():
            user_repr = user_repr.clone()
            user_repr[all_pad_rows] = 0.0
        t_emb = timestep_embedding_pi(t.reshape(-1), self.time_emb_dim)
        decoder_c_t = c_t
        propagated_c_t = None
        if self.decoder_relation_propagation != 'none':
            propagated_c_t = self._propagate_decoder_state(c_t)
            if self.decoder_relation_propagation == 'add':
                decoder_c_t = c_t + self.decoder_relation_propagation_weight * propagated_c_t
            elif self.decoder_relation_propagation == 'only':
                decoder_c_t = propagated_c_t
        decoder_inputs = [decoder_c_t]
        if self.decoder_relation_propagation == 'concat':
            decoder_inputs.append(propagated_c_t)
        if self.decoder_use_relation_scores:
            relation_channel = relation_scores if relation_scores is not None else torch.zeros_like(c_t)
            decoder_inputs.append(relation_channel)
        h = torch.cat(decoder_inputs + [user_repr, t_emb], dim=-1)
        mlp_score = self.decoder(h) if self.decoder_type in {'mlp', 'hybrid'} else None
        if self.decoder_type in {'itemwise', 'hybrid'}:
            itemwise_h = self.itemwise_context(torch.cat([user_repr, t_emb], dim=-1))
            itemwise_score = itemwise_h @ self.item_emb.weight.transpose(0, 1) + self.itemwise_bias
            x1_hat = itemwise_score if self.decoder_type == 'itemwise' else mlp_score + itemwise_score
        else:
            x1_hat = mlp_score
        if self.training and max_active > 1 and (self.sort_order != 'unordered'):
            valid_pairs = ~pad_mask[:, :-1] & ~pad_mask[:, 1:]
            pred_emb = self.aux_decoder(context[:, :-1, :])
            if self.aux_target_type == 'relation_conv':
                target_table = self._relation_convolved_item_embeddings()
                tgt_emb = target_table[sorted_items[:, 1:]].detach()
            else:
                tgt_emb = self.item_emb(sorted_items[:, 1:]).detach()
            if valid_pairs.any():
                aux_loss = F.mse_loss(pred_emb[valid_pairs], tgt_emb[valid_pairs])
            else:
                aux_loss = x1_hat.new_tensor(0.0)
            return (x1_hat, aux_loss)
        return x1_hat
