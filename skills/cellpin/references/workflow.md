# cellpin workflow: copy-paste code

Every block assumes the config cell has been run. Use a notebook (one block per cell) or a script
(one block after another), depending on what the user chose.

## 0. Install

```bash
pip install cellpin                      # Python >= 3.11
pip install "cellpin[spatial]"           # + SpatialData support
pip install spatialdata-io               # only to read raw Xenium/Atera output folders
```

Check the GPU: `python -c "import torch; print(torch.cuda.is_available())"`.

## 1. Config (all user-specific values go in one place)

```python
from pathlib import Path

import cellpin
import numpy as np
import scanpy as sc
import torch

SC_PATH = "/path/to/reference.h5ad"
SP_PATH = "/path/to/spatial.h5ad"
OUT_DIR = Path("./cellpin_output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYER = "counts"  # layer with RAW INTEGER counts, same name in both objects
BATCH_KEY = "batch"  # obs column in the reference (None if a single batch)
LABEL_KEY = "cell_type"  # obs column in the reference (None if no label transfer)
AREA_KEY = "cell_area"  # obs column in spatial data (None if absent)
N_HVG = 2000  # extra reference HVGs (targeted panels: 1000-2000; whole transcriptome: ~4000 total)
BATCH_SIZE = 512  # >30k reference cells: 512 (strongly recommended), very large atlases: 1024; <=30k: 256
EMBED_ONLY = False  # True -> fast path, no expression needed
```

## 2. Load

### 2a. AnnData (`.h5ad`)

```python
sc_adata = sc.read_h5ad(SC_PATH)
sp_adata = sc.read_h5ad(SP_PATH)
```

### 2b. Raw Xenium / Atera output folder

```python
import spatialdata_io

sdata = spatialdata_io.xenium(
    SP_PATH,
    transcripts=False,
    morphology_mip=False,
    morphology_focus=False,
    aligned_images=False,
    cells_labels=False,
    nucleus_labels=False,
    cells_boundaries=False,
    nucleus_boundaries=False,
)  # skip the heavy elements cellpin never uses; keep cells_boundaries=True if you want polygon plots
sp_adata = sdata.tables["table"]
sp_adata.obs_names = sp_adata.obs["cell_id"].astype(str)
```

(Atera uses the same `spatialdata_io.xenium` reader.)

### 2c. SpatialData `.zarr`

`setup_data`, `impute` and `label_transfer` all accept a `SpatialData` object directly, together with
`table_key="table"`. When `impute` gets a SpatialData object it returns the updated SpatialData.
For a first run it is simpler to work on the table: `sp_adata = sdata.tables["table"]`.

## 3. Prepare

### 3a. Put raw counts under `LAYER` in both objects

Find where the raw counts live **first**. The preflight script reports this. Common cases:

```python
# Spatial data straight from the reader: .X is raw counts
sp_adata.layers["counts"] = sp_adata.X.copy()

# CELLxGENE / many atlases: .X is log-normalized and the raw counts are in .raw.X
sc_adata.layers["counts"] = sc_adata.raw.X  # .raw has the same genes? check sc_adata.raw.shape
# If sc_adata.raw has MORE genes than sc_adata, rebuild: sc_adata = sc_adata.raw.to_adata(); then put .X in the layer

# Atlas stores counts under a different layer name
sc_adata.layers["counts"] = sc_adata.layers["count"]
```

### 3b. Gene identifiers

```python
# CELLxGENE: var_names are Ensembl IDs, symbols are in var["feature_name"]
sc_adata.var_names = sc_adata.var["feature_name"].astype(str)
sc_adata.var_names_make_unique()
sp_adata.var_names_make_unique()
```

Alternative: `setup_data(..., gene_symbols="feature_name")`. Note that this reads that column in
**both** objects, so the column must exist in both. Renaming `var_names` is usually simpler.

### 3c. Spatial QC + a measured normalized layer

```python
sc.pp.filter_cells(sp_adata, min_genes=10)  # targeted panels; Atera: min_genes=100 and min_counts≈250
# measured layer for comparisons later (area-normalized if cell_area exists, so it matches imputed_norm)
if AREA_KEY and AREA_KEY in sp_adata.obs:
    X = sp_adata.layers[LAYER]
    scale = (1e3 / sp_adata.obs[AREA_KEY].to_numpy(dtype=float))[:, None]
    norm = X.multiply(scale).tocsr() if hasattr(X, "multiply") else X * scale
    sp_adata.layers["log1p_norm"] = np.log1p(norm) if not hasattr(norm, "log1p") else norm.log1p()
else:
    norm = sc.pp.normalize_total(sp_adata, target_sum=1e3, layer=LAYER, inplace=False)["X"]
    sp_adata.layers["log1p_norm"] = sc.pp.log1p(norm, copy=True)
```

Do **not** overwrite `layers[LAYER]`. `.X` can be anything, because cellpin reads `layer=LAYER`.

### 3d. Reference gene space

**Targeted panel (Xenium, CosMx, MERSCOPE):** panel genes plus 1,000–2,000 batch-aware HVGs.

```python
panel = [g for g in sp_adata.var_names if g in sc_adata.var_names]
tmp = sc_adata.copy()
tmp.X = tmp.layers[LAYER].copy()
sc.pp.normalize_total(tmp, target_sum=1e4)
sc.pp.log1p(tmp)
sc.pp.highly_variable_genes(tmp, n_top_genes=N_HVG, batch_key=BATCH_KEY)
hvgs = tmp.var_names[tmp.var["highly_variable"]]
keep = sc_adata.var_names.isin(set(panel) | set(hvgs))
sc_adata = sc_adata[:, keep].copy()
del tmp
```

**Whole transcriptome (Atera):** ~4,000 reference HVGs, with the spatial data restricted to those that were measured.

```python
# compute HVGs on the reference as above with n_top_genes=4000, then:
sc_adata = sc_adata[:, sc_adata.var["highly_variable"]].copy()  # or the hvgs from tmp
genes = [g for g in sc_adata.var_names if g in sp_adata.var_names]
sp_adata = sp_adata[:, genes].copy()
```

If the reference is huge (more than ~300k cells) and memory or time is tight, subsample to around
100k–250k cells while keeping the batches represented: `sc.pp.subsample(sc_adata, n_obs=200_000)`.

### 3e. Memory note

`impute` copies every layer of `obs_adata` into the output as a dense matrix spanning the full
reference gene space. With many cells, drop layers you do not need before imputing
(keep `LAYER` and `log1p_norm`).

## 4. Preflight

```bash
python <skill-dir>/scripts/check_inputs.py --sc ref_prepared.h5ad --sp sp_prepared.h5ad \
    --layer counts --batch-key batch --label-key cell_type
```

Or in Python:

```python
import sys

sys.path.insert(0, "<skill-dir>/scripts")
from check_inputs import run_checks

ok = run_checks(sc_adata, sp_adata, layer=LAYER, batch_key=BATCH_KEY, label_key=LABEL_KEY, area_key=AREA_KEY)
assert ok, "Fix the FAIL items before training"
```

## 5. Setup + train

```python
sc_dataset, sp_dataset = cellpin.pp.setup_data(sc_adata, sp_adata, layer=LAYER, batch_key=BATCH_KEY)

model = cellpin.CellPin(sc_dataset)  # default config. Do not tune on the first run
model.fit(sc_dataset, pretrain_epochs=50, train_epochs=60, batch_size=BATCH_SIZE, save_checkpoints=False)
model.pl.losses()  # Stage 2 only; flat or jagged is fine
```

- `save_checkpoints=False` (the default) is the right choice. After `fit`, the model holds the **last** epoch's
  weights, and training is much faster without writing checkpoints.
- Early stopping is on by default (patience 10, monitoring `val_loss`).

### Checkpoints (only if the losses clearly diverge)

If `model.pl.losses()` (or `model.pl.losses(log_path=...)` for Stage 1) shows real divergence, such as a loss that
shoots up, `NaN`s, or a steady climb after an initial drop, retrain with checkpointing. `fit` then **automatically
restores the best (earlier) checkpoint** of each stage, meaning the epoch with the lowest monitored validation loss.
You don't need to load anything by hand:

```python
model = cellpin.CellPin(sc_dataset)
model.fit(sc_dataset, batch_size=BATCH_SIZE, save_checkpoints=True, output_dir=str(OUT_DIR / "ckpt"))
# prints e.g. "[train] Restored best checkpoint (epoch 23): best-epoch=23-val_loss=....ckpt"
```

The checkpoint files are in `ckpt/pretrain/checkpoints/` and `ckpt/train/checkpoints/`. If divergence keeps happening, also try a lower `lr`
(e.g. `config={"lr": 1e-4}`) or fewer epochs.

Expect roughly 10 minutes on one GPU for the example data, and longer for big atlases.

## 6. Impute / embed

```python
loader = torch.utils.data.DataLoader(sp_dataset, batch_size=1024, shuffle=False)

impute_kwargs = dict(mc_samples=1, mask_fraction=0.0) if EMBED_ONLY else {}
adata = model.impute(
    loader,
    obs_adata=sp_adata,  # same object as used in setup_data
    return_norm=True,  # adds layers["imputed_norm"]
    return_int=True,  # denoised integer counts; recommended
    area_key=AREA_KEY if AREA_KEY in sp_adata.obs else None,
    nb_count_samples=20,
    **impute_kwargs,
)
```

Output: `adata.X` and `layers["imputed"]` hold the imputed counts (full reference gene space),
`layers["imputed_norm"]` holds the log1p of the normalized imputed counts, `obsm["X_cellpin"]` holds
the embedding, and `var["is_measured"]` marks panel genes. The obs, obsm (including `spatial`) and
layers of `sp_adata` are copied over, and genes that were not measured are **filled with 0** in those layers.

## 7. Label transfer

> ⚠️ **Pass `sc_layer=LAYER` and `sp_layer=LAYER`.** When an object has no `obsm["X_cellpin"]` yet,
> `label_transfer` embeds it from `.X` by default. Most atlases keep log-normalized values in `.X`, which
> gives a wrong reference embedding. `sc_layer`/`sp_layer` make it read the raw-count layer instead.

```python
acc, adata = cellpin.tl.label_transfer(
    model,
    sc_adata,
    LABEL_KEY,
    adata,
    conf_threshold=0.5,  # cells below this get "Unknown"; use 0.0 to label everything
    k=15,
    sc_layer=LAYER,  # raw counts of the reference (None = .X)
    sp_layer=LAYER,  # raw counts of the spatial data (None = .X)
)
print(f"Reference held-out accuracy: {acc:.3f}  (sanity check only)")
adata.obs["cellpin_annotation"].value_counts()
```

- Start with a **coarse** label column. Move to finer labels only if the reference supports them.
- Every label class needs at least 2 reference cells (stratified split), and realistically 20 or more.
- If the reference is poorly annotated, run Leiden on the cellpin graph and annotate the clusters by marker genes instead.
- On cellpin ≤ 0.3.0 (no `sc_layer`): set the reference embedding first with
  `sc_adata.obsm["X_cellpin"] = model.get_cell_embedding(torch.utils.data.DataLoader(sc_dataset, batch_size=1024, shuffle=False))`.

## 8. Validate

```python
sc.pp.neighbors(adata, use_rep="X_cellpin")
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=1.0, flavor="igraph")

MARKERS = ["CD3E", "MS4A1", "EPCAM", "PTPRC"]  # pick markers for this tissue
measured = [g for g in MARKERS if g in adata.var_names and adata.var.loc[g, "is_measured"]]

sc.pl.umap(adata, color=["leiden"] + (["cellpin_annotation"] if "cellpin_annotation" in adata.obs else []))
sc.pl.umap(adata, color=measured, layer="log1p_norm", vmax="p99", title=[f"Measured {g}" for g in measured])
sc.pl.umap(adata, color=measured, layer="imputed_norm", vmax="p99", title=[f"Imputed {g}" for g in measured])
if "spatial" in adata.obsm:
    sc.pl.spatial(adata, color=measured, layer="imputed_norm", spot_size=15, vmax="p99")
```

What "good" looks like:
- Known cell types form separate clusters, and markers in the **measured** layer concentrate in the matching clusters.
- Imputed markers agree in space with measured markers of the same cell type.
- Transferred labels match marker expression and the expected tissue architecture.

Optional quantitative check: the per-gene Pearson r between `layers[LAYER]` and `layers["imputed"]` over
measured genes. Around 0.6 was typical in the tutorial. For an honest imputation benchmark, hold out
about 10% of panel genes from `sp_adata` before `setup_data`, then correlate their imputed values with the
true counts (basic tutorial, section B).

## 9. Save

```python
adata.write_h5ad(OUT_DIR / "cellpin_output.h5ad")
model.save(OUT_DIR / "cellpin_model.pt")
```

For a SpatialData input, `sdata.write(OUT_DIR / "cellpin_output.zarr")`.

## Tuning (encouraged if the results are unsatisfying)

The defaults work robustly across tissues and platforms, so always start with them. If validation (§8) is
unsatisfying, try tuning. Change one thing at a time and compare against the default run using the same checks.

| Symptom | Try |
|---|---|
| Over-smoothed reconstructions, same-type cells look identical | KL floor: `config={"kl_free_bits": 2.0}` **plus** `fit(..., checkpoint_monitor="val_reconst_loss")`, see below |
| Unstable training, or a reference with more than 30k cells | `batch_size=512` (very large atlases: `1024`) |
| Batch structure visible in the embedding | make sure `batch_key` is set and points to the right column |
| Loss diverges | lower `lr` (e.g. `1e-4`), see "Checkpoints" above |
| Still unsatisfying | learning rate (`lr`), `pretrain_epochs`/`train_epochs`, loss weights (`lambda_recon`, `kl_weight`, `lambda_snn`, `lambda_distill`, ...), `distillation_mode`. All are listed in the [API reference](https://cellpin.readthedocs.io/en/latest/api.html) and `configs/cellpin_config.yaml` |

Config keys are passed as `cellpin.CellPin(sc_dataset, config={...})`. Unspecified keys keep their defaults.

### Sharper reconstructions: the KL floor (`kl_free_bits`)

Opt-in and **not a default**. It applies a per-latent-dimension free-bits floor to the KL term. In our tests, `2.0`
sharpened panel-gene reconstruction noticeably without hurting accuracy, but the best value depends on the dataset,
so a small sweep (e.g. 1.0 / 2.0 / 4.0) is worthwhile.

```python
model = cellpin.CellPin(sc_dataset, config={"kl_free_bits": 2.0})
model.fit(
    sc_dataset,
    batch_size=BATCH_SIZE,
    checkpoint_monitor="val_reconst_loss",  # REQUIRED with the KL floor
)
```

Why monitor the reconstruction loss: with the floor, the KL part of `val_loss` cannot drop below the floor, so
`val_loss` stops being a good signal for early stopping. `val_reconst_loss` tracks what you actually care about.
