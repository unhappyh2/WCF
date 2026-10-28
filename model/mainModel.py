import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.utils import InputType
from model.module.FlowCFModule import TransformerFlowModel, mean_flat

class mainModel(GeneralRecommender, AutoEncoderMixin):
    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        super().build_histroy_items(dataset)
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_steps = config['n_steps'] if 'n_steps' in config else 200
        self.s_steps = config['s_steps'] if 's_steps' in config else 20
        self.time_emb_size = config['time_embedding_size'] if 'time_embedding_size' in config else 64
        self.dims_mlp = config['dims_mlp'] if 'dims_mlp' in config else [300]
        self.flow_act_func = config['flow_act_func'] if 'flow_act_func' in config else 'tanh'
        self.flow_dropout = config['flow_dropout'] if 'flow_dropout' in config else 0.3
        self.observation_mode = config['observation_mode'] if 'observation_mode' in config else 'bernoulli'
        self.training_objective = config['training_objective'] if 'training_objective' in config else 'flow'
        if self.training_objective not in {'flow', 'endpoint'}:
            raise ValueError(f'Unsupported training_objective: {self.training_objective}')
        self.path_type = config['path_type'] if 'path_type' in config else 'weibull'
        self.weibull_lambda_min = config['weibull_lambda_min'] if 'weibull_lambda_min' in config else 0.35
        self.weibull_lambda_max = config['weibull_lambda_max'] if 'weibull_lambda_max' in config else 1.5
        self.weibull_k = config['weibull_k'] if 'weibull_k' in config else 1.5
        self.relation_lambda_power = config['relation_lambda_power'] if 'relation_lambda_power' in config else 1.0
        self.power_alpha_min = config['power_alpha_min'] if 'power_alpha_min' in config else 0.5
        self.power_alpha_max = config['power_alpha_max'] if 'power_alpha_max' in config else 2.0
        self.gaussian_sigma = config['gaussian_sigma'] if 'gaussian_sigma' in config else 0.25
        self.gaussian_mu_max = config['gaussian_mu_max'] if 'gaussian_mu_max' in config else 0.6
        self.history_projection_gamma = config['history_projection_gamma'] if 'history_projection_gamma' in config else 0.0
        self.x0_mask_rate = config['x0_mask_rate'] if 'x0_mask_rate' in config else 0.0
        self.x0_source = config['x0_source'] if 'x0_source' in config else 'history_mask' if self.x0_mask_rate > 0.0 else 'popularity'
        self.relation_context_source = config['relation_context_source'] if 'relation_context_source' in config else 'x0'
        self.relation_context_mask_rate = config['relation_context_mask_rate'] if 'relation_context_mask_rate' in config else self.x0_mask_rate
        self.inference_x0_source = config['inference_x0_source'] if 'inference_x0_source' in config else 'history_full'
        self.inference_relation_context_source = config['inference_relation_context_source'] if 'inference_relation_context_source' in config else 'history_full'
        self.loss_pop_weight = config['loss_pop_weight'] if 'loss_pop_weight' in config else False
        self.vae_beta = config['vae_beta'] if 'vae_beta' in config else 0.001
        self.aux_loss_weight = config['aux_loss_weight'] if 'aux_loss_weight' in config else 0.1
        flow_model_type = config['flow_model_type'] if 'flow_model_type' in config else 'mlp'
        self.sort_order = config['sort_order'] if 'sort_order' in config else 'hot_cold'
        self.relation_graph_diag_weight = config['relation_graph_diag_weight'] if 'relation_graph_diag_weight' in config else 0.0
        self.relation_graph_mode = config['relation_graph_mode'] if 'relation_graph_mode' in config else 'cooccurrence'
        self.relation_graph_seed = config['seed'] if 'seed' in config else 2026
        self.decoder_relation_propagation = config['decoder_relation_propagation'] if 'decoder_relation_propagation' in config else 'none'
        self.item_frequencies = self._get_item_frequencies()
        self.item_popularity_norm = self._normalize_item_popularity(self.item_frequencies)
        if self.path_type == 'relation_weibull' or self.sort_order in {'relation_desc', 'relation_asc'} or self.decoder_relation_propagation != 'none':
            self.item_item_graph = self._build_item_item_graph(dataset)
        else:
            self.item_item_graph = None
        self.time_steps = torch.linspace(0, 1, self.n_steps + 1)
        max_seq_len = config['max_seq_len'] if 'max_seq_len' in config else 50
        transformer_d_model = config['transformer_d_model'] if 'transformer_d_model' in config else 64
        transformer_nhead = config['transformer_nhead'] if 'transformer_nhead' in config else 4
        transformer_ff_multiplier = config['transformer_ff_multiplier'] if 'transformer_ff_multiplier' in config else 2
        transformer_num_layers = config['transformer_num_layers'] if 'transformer_num_layers' in config else 2
        use_causal_mask = config['use_causal_mask'] if 'use_causal_mask' in config else True
        use_pos_emb = config['use_pos_emb'] if 'use_pos_emb' in config else False
        use_relation_value_emb = config['use_relation_value_emb'] if 'use_relation_value_emb' in config else False
        relation_film_type = config['relation_film_type'] if 'relation_film_type' in config else 'none'
        transformer_decoder_type = config['transformer_decoder_type'] if 'transformer_decoder_type' in config else 'mlp'
        context_encoder_type = config['context_encoder_type'] if 'context_encoder_type' in config else 'transformer'
        decoder_use_relation_scores = config['decoder_use_relation_scores'] if 'decoder_use_relation_scores' in config else False
        decoder_relation_propagation_weight = config['decoder_relation_propagation_weight'] if 'decoder_relation_propagation_weight' in config else 1.0
        aux_target_type = config['aux_target_type'] if 'aux_target_type' in config else 'item'
        aux_relation_conv_weight = config['aux_relation_conv_weight'] if 'aux_relation_conv_weight' in config else 0.0
        self.flow_model = TransformerFlowModel(n_items=self.n_items, time_emb_size=self.time_emb_size, item_popularity_norm=self.item_popularity_norm, item_relation_graph=self.item_item_graph, dropout=self.flow_dropout, max_seq_len=max_seq_len, d_model=transformer_d_model, nhead=transformer_nhead, dim_feedforward_multiplier=transformer_ff_multiplier, num_layers=transformer_num_layers, use_causal_mask=use_causal_mask, sort_order=self.sort_order, use_pos_emb=use_pos_emb, use_relation_value_emb=use_relation_value_emb, relation_film_type=relation_film_type, decoder_type=transformer_decoder_type, context_encoder_type=context_encoder_type, decoder_hidden_dims=self.dims_mlp, decoder_use_relation_scores=decoder_use_relation_scores, decoder_relation_propagation=self.decoder_relation_propagation, decoder_relation_propagation_weight=decoder_relation_propagation_weight, aux_target_type=aux_target_type, aux_relation_conv_weight=aux_relation_conv_weight, observation_mode=self.observation_mode)

    def _get_item_frequencies(self) -> torch.Tensor:
        item_counts = torch.zeros(self.n_items, device=self.device)
        for user_id in range(self.n_users):
            ids = self.history_item_id[user_id]
            item_counts[ids] += 1
        item_counts[0] = 0.0
        return item_counts / max(self.n_users, 1)

    def _normalize_item_popularity(self, item_frequencies: torch.Tensor) -> torch.Tensor:
        real = item_frequencies[1:]
        pop_min = real.min()
        pop_max = real.max()
        pop_norm = torch.zeros_like(item_frequencies)
        pop_norm[1:] = (real - pop_min) / (pop_max - pop_min + 1e-08)
        return pop_norm

    @staticmethod
    def _build_item_item_graph_from_matrix(inter_matrix: sp.spmatrix, device: torch.device, diag_weight: float=0.0) -> torch.Tensor:
        inter_matrix = inter_matrix.tocsr().astype(np.float32)
        user_degree = np.asarray(inter_matrix.sum(axis=1)).reshape(-1)
        item_degree = np.asarray(inter_matrix.sum(axis=0)).reshape(-1)
        user_inv_sqrt = np.zeros_like(user_degree, dtype=np.float32)
        item_inv_sqrt = np.zeros_like(item_degree, dtype=np.float32)
        user_mask = user_degree > 0
        item_mask = item_degree > 0
        user_inv_sqrt[user_mask] = np.power(user_degree[user_mask], -0.5)
        item_inv_sqrt[item_mask] = np.power(item_degree[item_mask], -0.5)
        norm_inter = inter_matrix.multiply(user_inv_sqrt[:, None])
        norm_inter = norm_inter.multiply(item_inv_sqrt[None, :])
        graph = (norm_inter.T @ norm_inter).tocoo().astype(np.float32)
        if diag_weight == 0.0:
            graph.setdiag(0.0)
        elif diag_weight != 1.0:
            graph.setdiag(graph.diagonal() * diag_weight)
        graph.eliminate_zeros()
        indices = torch.LongTensor(np.vstack([graph.row, graph.col]))
        values = torch.FloatTensor(graph.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(graph.shape), device=device).coalesce()

    def _build_item_item_graph(self, dataset) -> torch.Tensor:
        mode = str(getattr(self, 'relation_graph_mode', 'cooccurrence'))
        if mode == 'identity':
            indices = torch.arange(self.n_items, device=self.device)
            values = torch.ones(self.n_items, device=self.device)
            return torch.sparse_coo_tensor(torch.stack([indices, indices]), values, torch.Size([self.n_items, self.n_items]), device=self.device).coalesce()
        inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        graph = self._build_item_item_graph_from_matrix(inter_matrix, device=self.device, diag_weight=float(self.relation_graph_diag_weight))
        if mode == 'cooccurrence':
            return graph
        if mode == 'random_topology':
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(getattr(self, 'relation_graph_seed', 2026)))
            permutation = torch.arange(self.n_items, dtype=torch.long)
            permutation[1:] = torch.randperm(self.n_items - 1, generator=generator) + 1
            permutation = permutation.to(graph.device)
            return torch.sparse_coo_tensor(permutation[graph.indices()], graph.values(), graph.shape, device=graph.device).coalesce()
        if mode == 'random':
            graph = graph.coalesce()
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(getattr(self, 'relation_graph_seed', 2026)))
            values = torch.rand(graph.values().numel(), generator=generator, dtype=graph.values().dtype)
            values = values.to(device=graph.values().device)
            return torch.sparse_coo_tensor(graph.indices(), values, graph.shape, device=graph.device).coalesce()
        raise ValueError(f'Unsupported relation_graph_mode: {mode}')

    def _row_normalize_relation(self, relation_scores: torch.Tensor) -> torch.Tensor:
        relation_scores = relation_scores.clamp(min=0.0)
        if relation_scores.size(1) > 0:
            relation_scores = relation_scores.clone()
            relation_scores[:, 0] = 0.0
        row_min = relation_scores.min(dim=1, keepdim=True).values
        row_max = relation_scores.max(dim=1, keepdim=True).values
        return (relation_scores - row_min) / (row_max - row_min + 1e-08)

    def _build_user_item_relation(self, x0: torch.Tensor) -> torch.Tensor:
        if self.item_item_graph is None:
            raise ValueError('item_item_graph is required for relation-based path or sorting')
        graph = self.item_item_graph.to(x0.device)
        if graph.is_sparse:
            relation_scores = torch.sparse.mm(graph, x0.transpose(0, 1)).transpose(0, 1)
        else:
            relation_scores = torch.matmul(x0, graph.transpose(0, 1))
        return self._row_normalize_relation(relation_scores)

    def _sample_x0(self, batch_size: int, device: torch.device, x1: torch.Tensor=None):
        x0_source = getattr(self, 'x0_source', 'history_mask' if self.x0_mask_rate > 0.0 else 'popularity')
        if x0_source == 'history_mask' and self.x0_mask_rate > 0.0 and (x1 is not None):
            keep_mask = torch.bernoulli(torch.full_like(x1, 1.0 - self.x0_mask_rate))
            return (x1 * keep_mask, keep_mask)
        if x0_source != 'popularity':
            raise ValueError(f'Unsupported x0_source: {x0_source}')
        return (torch.bernoulli(self.item_frequencies.to(device).expand(batch_size, -1)), None)

    def _sample_relation_context(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        source = getattr(self, 'relation_context_source', 'x0')
        if source == 'x0':
            return x0
        if source == 'history_mask':
            mask_rate = float(getattr(self, 'relation_context_mask_rate', self.x0_mask_rate))
            keep_mask = torch.bernoulli(torch.full_like(x1, 1.0 - mask_rate))
            return x1 * keep_mask
        if source == 'history_full':
            return x1
        raise ValueError(f'Unsupported relation_context_source: {source}')

    def _build_inference_x0(self, x_history: torch.Tensor) -> torch.Tensor:
        source = getattr(self, 'inference_x0_source', 'history_full')
        if source == 'history_full':
            return x_history
        if source == 'popularity':
            return torch.bernoulli(self.item_frequencies.to(x_history.device).expand_as(x_history))
        if source == 'popularity_mean':
            return self.item_frequencies.to(x_history.device).expand_as(x_history)
        raise ValueError(f'Unsupported inference_x0_source: {source}')

    def _build_inference_relation_context(self, x_history: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        source = getattr(self, 'inference_relation_context_source', 'history_full')
        if source == 'history_full':
            return x_history
        if source == 'x0':
            return x0
        raise ValueError(f'Unsupported inference_relation_context_source: {source}')

    def _inference_start_step(self) -> int:
        source = getattr(self, 'inference_x0_source', 'history_full')
        if source == 'history_full':
            return max(self.n_steps - self.s_steps, 0)
        return 0

    def _build_path_gamma(self, t: torch.Tensor, pop_norm: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.to(device=pop_norm.device, dtype=dtype)
        pop_norm = pop_norm.to(dtype=dtype)
        if self.path_type == 'linear':
            return t.expand_as(pop_norm)
        elif self.path_type == 'weibull':
            lambda_i = self.weibull_lambda_min + (self.weibull_lambda_max - self.weibull_lambda_min) * (1.0 - pop_norm)
            t_safe = t.clamp(0.0, 1.0)
            k = torch.as_tensor(self.weibull_k, dtype=dtype, device=pop_norm.device)
            raw_gamma = 1.0 - torch.exp(-torch.pow(t_safe / lambda_i.clamp(min=1e-06), k))
            end_gamma = 1.0 - torch.exp(-torch.pow(1.0 / lambda_i.clamp(min=1e-06), k))
            return (raw_gamma / end_gamma.clamp(min=1e-08)).clamp(0.0, 1.0)
        elif self.path_type == 'power':
            alpha_i = self.power_alpha_min + (self.power_alpha_max - self.power_alpha_min) * (1.0 - pop_norm)
            t_safe = t.clamp(1e-08, 1.0)
            return torch.pow(t_safe, alpha_i).clamp(0.0, 1.0)
        elif self.path_type == 'gaussian':
            import math
            mu_i = self.gaussian_mu_max * (1.0 - pop_norm)
            t_safe = t.clamp(0.0, 1.0)
            sigma_t = torch.as_tensor(self.gaussian_sigma, dtype=dtype, device=pop_norm.device)
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            phi = lambda x: 0.5 * (1.0 + torch.erf(x * inv_sqrt2 / sigma_t))
            raw = phi(t_safe - mu_i) - phi(-mu_i)
            norm = phi(1.0 - mu_i) - phi(-mu_i)
            return (raw / norm.clamp(min=1e-08)).clamp(0.0, 1.0)
        else:
            raise ValueError(f'Unsupported path_type: {self.path_type}')

    def _build_relation_path_gamma(self, t: torch.Tensor, relation_scores: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.to(device=relation_scores.device, dtype=dtype)
        relation_scores = relation_scores.to(dtype=dtype)
        relation_lambda_power = float(getattr(self, 'relation_lambda_power', 1.0))
        relation_power = 1.0 - torch.pow(1.0 - relation_scores.clamp(0.0, 1.0), relation_lambda_power)
        lambda_ui = self.weibull_lambda_max - (self.weibull_lambda_max - self.weibull_lambda_min) * relation_power
        t_safe = t.clamp(0.0, 1.0)
        k = torch.as_tensor(self.weibull_k, dtype=dtype, device=relation_scores.device)
        raw_gamma = 1.0 - torch.exp(-torch.pow(t_safe / lambda_ui.clamp(min=1e-06), k))
        end_gamma = 1.0 - torch.exp(-torch.pow(1.0 / lambda_ui.clamp(min=1e-06), k))
        return (raw_gamma / end_gamma.clamp(min=1e-08)).clamp(0.0, 1.0)

    def _build_x_t(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, relation_scores: torch.Tensor=None) -> torch.Tensor:
        if self.path_type == 'relation_weibull':
            if relation_scores is None:
                relation_scores = self._build_user_item_relation(x0)
            gamma = self._build_relation_path_gamma(t, relation_scores, x1.dtype)
        else:
            pop_norm = self.item_popularity_norm.to(x1.device).unsqueeze(0).expand_as(x1)
            gamma = self._build_path_gamma(t, pop_norm, x1.dtype)
        x_t = ((1 - gamma) * x0 + gamma * x1).clamp(0.0, 1.0)
        if self.history_projection_gamma > 0:
            t_coef = t if t.dim() == 2 else t.unsqueeze(1)
            history_floor = (self.history_projection_gamma * t_coef * x1).clamp(0.0, 1.0)
            x_t = torch.max(x_t, history_floor)
        return x_t

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, relation_scores: torch.Tensor=None):
        if relation_scores is not None and isinstance(self.flow_model, TransformerFlowModel):
            return self.flow_model(x_t, t, relation_scores=relation_scores)
        return self.flow_model(x_t, t)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        x1 = self.get_rating_matrix(user)
        x0, keep_mask = self._sample_x0(x1.size(0), x1.device, x1)
        path_type = getattr(self, 'path_type', 'weibull')
        sort_order = getattr(self, 'sort_order', 'hot_cold')
        relation_context = self._sample_relation_context(x0, x1)
        relation_scores = self._build_user_item_relation(relation_context) if path_type == 'relation_weibull' or sort_order in {'relation_desc', 'relation_asc'} else None
        if getattr(self, 'training_objective', 'flow') == 'endpoint':
            t = torch.ones((x1.size(0), 1), device=x1.device, dtype=x1.dtype)
            x_t = x0
        else:
            steps = torch.randint(0, self.n_steps, (x1.size(0),), device=self.device)
            t = self.time_steps.to(x1.device)[steps].unsqueeze(1)
            x_t = self._build_x_t(x0, x1, t, relation_scores=relation_scores)
        out = self.forward(x_t, t.squeeze(-1), relation_scores=relation_scores)
        if isinstance(out, tuple):
            if len(out) == 3:
                x1_hat, mu, logvar = out
                kl = -0.5 * mean_flat(1 + logvar - mu.pow(2) - logvar.exp()).mean()
                aux_loss = 0.0
            else:
                x1_hat, aux_loss = out
                kl = 0.0
        else:
            x1_hat, kl, aux_loss = (out, 0.0, 0.0)
        sq_err = (x1 - x1_hat) ** 2
        if self.loss_pop_weight:
            inv_pop = (1.0 - self.item_popularity_norm).unsqueeze(0).to(x1.device)
            sq_err = sq_err * inv_pop
        if keep_mask is not None:
            loss_weight = 1.0 - x1 * keep_mask
            recon = mean_flat(loss_weight * sq_err).mean()
        else:
            recon = mean_flat(sq_err).mean()
        return recon + self.vae_beta * kl + self.aux_loss_weight * aux_loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        x_bar = self.get_rating_matrix(user)
        xt = self._build_inference_x0(x_bar)
        path_type = getattr(self, 'path_type', 'weibull')
        sort_order = getattr(self, 'sort_order', 'hot_cold')
        relation_context = self._build_inference_relation_context(x_bar, xt)
        relation_scores = self._build_user_item_relation(relation_context) if path_type == 'relation_weibull' or sort_order in {'relation_desc', 'relation_asc'} else None
        if getattr(self, 'training_objective', 'flow') == 'endpoint':
            t = torch.ones(xt.shape[0], device=x_bar.device, dtype=x_bar.dtype)
            out = self.forward(xt, t, relation_scores=relation_scores)
            return out[0] if isinstance(out, tuple) else out
        start_step = self._inference_start_step()
        for i_t in range(start_step, self.n_steps):
            t = self.time_steps[i_t].repeat(xt.shape[0], 1).to(x_bar.device)
            out = self.forward(xt, t.squeeze(-1), relation_scores=relation_scores)
            x1_hat = out[0] if isinstance(out, tuple) else out
            if i_t == self.n_steps - 1:
                break
            t_next = self.time_steps[i_t + 1].repeat(xt.shape[0], 1).to(x_bar.device)
            v = (x1_hat - xt) / (1 - t)
            xt_pos_probs = xt + v * (t_next - t)
            xt_probs = torch.stack([1 - xt_pos_probs, xt_pos_probs], dim=-1)
            xt = xt_probs.argmax(dim=-1).float()
        return x1_hat

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        scores = self.full_sort_predict(interaction)
        return scores.gather(1, item.view(-1, 1)).squeeze(1)
