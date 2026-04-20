# CellPin

CellPin is a two-stage VAE for imputing full-gene expression in spatial transcriptomics data using a single-cell RNA-seq reference. A panel encoder (seeing only the measured spatial genes) is trained to match the latent geometry of a full-gene encoder, then used at inference to decode complete expression profiles.

See the [tutorial notebook](cellpin_tutorial.ipynb) for a step-by-step walkthrough.

---

## Installation

### PyTorch prerequisite

CellPin requires PyTorch, which must be installed separately to match your CUDA version. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

Example for CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### pip (from source)

```bash
git clone https://github.com/Saur-Lab/cellpin.git
cd cellpin
pip install -e .
```

### uv (recommended)

```bash
git clone https://github.com/Saur-Lab/cellpin.git
cd cellpin
uv sync
```

`uv` picks up the `pyproject.toml` automatically; no separate `pip install` is needed. Activate the environment with `source .venv/bin/activate` or prefix commands with `uv run`.

---

## Quick start

```python
import torch
import scanpy as sc
import cellpin

# Load your data
sc_adata = sc.read_h5ad("sc_reference.h5ad")   # single-cell RNA-seq atlas
sp_adata = sc.read_h5ad("spatial.h5ad")        # spatial transcriptomics (panel genes only)

# Align genes and build datasets
# Panel genes = intersection of sc and spatial gene sets
sc_dataset, st_dataset = cellpin.pp.setup(sc_adata, sp_adata, layer="counts")

# Build model and train (Stage 1: pretrain + Stage 2: panel encoder)
model = cellpin.CellPin(sc_dataset, config="configs/cellpin_config.yaml")
model.fit(sc_dataset, pretrain_epochs=50, train_epochs=60)

# Impute full-gene expression for spatial cells
dl = torch.utils.data.DataLoader(st_dataset, batch_size=512, shuffle=False)
adata_imputed = model.impute(
    dl,
    obs_adata=sp_adata,
    return_norm=True,
    area_key="cell_area",   # set to None if no cell_area column → library-size normalisation
    nb_count_samples=20,
)

# adata_imputed.X                   — imputed counts
# adata_imputed.obsm["X_cellpin"]   — cell embeddings
# adata_imputed.layers["imputed"]   — imputed counts
# adata_imputed.layers["imputed_norm"]  — normalised imputed counts (log1p)
adata_imputed.write_h5ad("cellpin_imputed.h5ad")
```

### Batch correction (multi-sample atlas)

If your scRNA reference spans multiple samples or donors, pass a `batch_key` to `setup()`. CellPin will encode batch as a categorical covariate and condition the decoder on it. During spatial inference (where no batch label is available) the decoder is conditioned on a uniform soft one-hot over all batches.

```python
# sc_adata.obs["sample_id"] holds the batch labels (e.g. donor ID, sample name)
sc_dataset, st_dataset = cellpin.pp.setup(
    sc_adata, sp_adata,
    layer="counts",
    batch_key="sample_id",   # any obs column with categorical batch labels
)

# Training and imputation are unchanged — batch correction is automatic
model = cellpin.CellPin(sc_dataset, config="configs/cellpin_config.yaml")
model.fit(sc_dataset, pretrain_epochs=50, train_epochs=60)

dl = torch.utils.data.DataLoader(st_dataset, batch_size=512, shuffle=False)
adata_imputed = model.impute(dl, obs_adata=sp_adata, return_norm=True, nb_count_samples=20)
```

---

## Scripts

The `scripts/` directory provides ready-to-run entry points.

### Basic training and imputation

```bash
python scripts/run_cellpin.py \
    --adata_path sc_reference.h5ad \
    --spatial_path spatial.h5ad \
    --output_dir experiments/cellpin_run
```

Output: `experiments/cellpin_run/cellpin_imputed.h5ad`



### Key arguments (all scripts)

| Argument | Default | Description |
|---|---|---|
| `--layer` | `counts` | Expression layer to use (empty string → `.X`) |
| `--pretrain_epochs` | `50` | Epochs for Stage 1 |
| `--train_epochs` | `60` | Epochs for Stage 2 |
| `--batch_size` | `256` | Mini-batch size |
| `--freeze_pretrained` | `False` | Freeze full-gene encoder during Stage 2 |
| `--precision` | `16-mixed` | PyTorch Lightning precision |
| `--devices` | `[0]` | GPU device IDs |
| `--seed` | `42` | Random seed |

---

## Configuration

Architecture and loss hyperparameters live in `configs/cellpin_config.yaml`. Defaults:

```yaml
n_latent: 128
n_hidden: 1024
encoder_layers: 7
decoder_layers: 2
reconstruction_loss: nb      # nb | zinb | poisson | normal | zin
kl_weight: 0.08
lambda_recon: 1.17
lambda_inv: 20
lambda_snn: 0.085
encoder_noise_std: 0.1
panel_mixup_alpha: 0.1
poisson_resample_rate: 0.1
```

Pass a custom config with `--config path/to/config.yaml`, or supply a dict directly to `CellPin(sc_dataset, config={...})`.

---

## Imputation API

Use `model.impute()` for full-featured inference (recommended):

```python
adata_out = model.impute(
    dataloader,
    obs_adata=sp_adata,        # copies .obs to output
    mc_samples=50,             # stochastic forward passes for MC averaging
    mask_fraction=0.2,         # fraction of panel genes zeroed per MC pass
    return_norm=True,          # adds layers["imputed_norm"]
    norm_target_sum=1000.0,    # total-count normalisation target
    area_key="cell_area",      # obs column for area-based normalisation; None → library-size
    nb_count_samples=20,       # NB draws inside log1p to correct Jensen bias
    return_int=False,          # True → round X to integer counts
)
```

`impute_to_anndata()` is a simpler alternative that skips MC averaging and returns raw mean counts + embeddings.

---

## UMAP from CellPin embeddings

After imputation, cell embeddings are stored in `adata_imputed.obsm["X_cellpin"]`. Use them directly as input for neighborhood graph construction and UMAP:

```python
sc.pp.neighbors(adata_imputed, use_rep="X_cellpin")
sc.tl.umap(adata_imputed)
sc.pl.umap(adata_imputed, color="cell_type")
```

---

## Development

```bash
# Install with dev/test extras
pip install -e ".[dev,test]"

# Run tests
pytest

# Lint / format
ruff check src/
ruff format src/
```
