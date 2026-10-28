# WCF

Implementation of **WCF: Relation-Guided Collaborative Flow with Graph-Propagated Reconstruction for Recommendation**, accepted by Expert Systems with Applications.

DOI: https://doi.org/10.1016/j.eswa.2026.134787

WCF combines relation-guided Weibull paths, relation-ranked Transformer interpretation of intermediate active states, and graph-propagated reconstruction.

## Setup

Use Python 3.10 and install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Datasets are not distributed in this repository. Prepare a RecBole atomic interaction file at `dataset/ml-100k/ml-100k.inter`. The default configuration expects tab-separated columns with this header:

```text
user_id:token	item_id:token	rating:float	timestamp:float
```

Edit `dataset`, `data_path`, field settings, and filtering rules in `config.yaml` for your local data. The supplied configuration is the current ML-100K configuration; it is not a set of tuned settings for every dataset. Relations are constructed from the training split only.

## Train and evaluate

```bash
python run.py --config config.yaml
```

For CPU execution, set `use_gpu: False`. For checkpoint evaluation, set `do_train: False` and run:

```bash
python run.py --config config.yaml --checkpoint saved/model.pth
```

## Files

- `run.py`: RecBole training and evaluation entry point.
- `config.yaml`: model, data, and evaluation settings.
- `model/mainModel.py`: training relations, path construction, loss, and inference.
- `model/module/FlowCFModule.py`: intermediate-state Transformer and decoder.
- `utils/flowcf_dataloader.py`: self-contained RecBole user-wise data loader registration.

Only the WCF backbone is included. Experiment queues, baseline runners, plotting scripts, datasets, checkpoints, and paper files are excluded. Internal configuration switches are retained for compatibility with the research implementation.
