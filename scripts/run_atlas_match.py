import anndata as ad
import cellpin
import scanpy as sc
import torch

# ── 1. Load data ──────────────────────────────────────────────────────────────
LAYER         = "counts"
CELL_TYPE_COL = "Level_4"
ATLAS_EMB_KEY = "x_scVI_1"
OUT_PATH      = "/mnt/storage/philipp/PP_FlexResource/public/spatial/combined_coembedding.h5ad"

sc_adata = sc.read_h5ad("/mnt/storage/philipp/PP_FlexResource/public/core/Core_annotated_v2_with_Level_5_harmonized.h5ad")
sp_adata = sc.read_h5ad("/mnt/storage/philipp/PP_FlexResource/public/spatial/spatial_all_samples_inner_cellpose.h5ad")
print(sc_adata)
print(f"scRNA  : {sc_adata.n_obs:,} cells × {sc_adata.n_vars:,} genes")
print(f"Spatial: {sp_adata.n_obs:,} cells × {sp_adata.n_vars:,} genes")
print(f"Atlas embedding dim: {sc_adata.obsm[ATLAS_EMB_KEY].shape[1]}")
sp_adata.layers["counts"] = sp_adata.X
# ── 2. Align gene spaces & build datasets ─────────────────────────────────────
sc_dataset, sp_dataset = cellpin.pp.setup_data(
    sc_adata, sp_adata,
    layer=LAYER,
)

# ── 3. Build model ────────────────────────────────────────────────────────────
config = {
    # network architecture
    "atlas_hidden": 1024,
    "atlas_blocks": 8,
    "atlas_expansion": 2.0,
    "atlas_dropout": 0.1,
    "atlas_drop_path_rate": 0.1,
    # augmentation (simulate lower spatial capture efficiency)
    "poisson_resample_rate": 0.4,
    "spatial_resample_rate": 0.85,
    "panel_mixup_alpha": 0.3,
    # loss weights
    "atlas_distill_weight": 1.0,
    "atlas_consistency_weight": 1.0,
    "atlas_cos_weight": 0.1,
    # training schedule
    "atlas_aug_warmup_frac": 0.10,
    "atlas_lr_warmup_epochs": 5,
    "atlas_ema_decay": 0.999,
}

model = cellpin.CellPin(sc_dataset, config=config)

# ── 4. Train atlas-matching network ───────────────────────────────────────────
model.match_emb(
    sc_dataset,
    emb_key=ATLAS_EMB_KEY,
    train_epochs=60,
    batch_size=1024,
    early_stopping_patience=20,
    checkpoint_monitor="val_knn_overlap",
    early_stopping_mode="max",
    accelerator="gpu",
    devices=1,
)

# ── 5. Embed spatial → atlas space & label transfer (pre-finetune) ────────────
sp_dl = torch.utils.data.DataLoader(sp_dataset, batch_size=512, shuffle=False, num_workers=4)
sp_adata.obsm["X_cellpin_match"] = model.embed_atlas(sp_dl)

acc_before, sp_adata = model.tl.label_transfer(sc_adata, CELL_TYPE_COL, sp_adata)
print(f"match_emb (pre-finetune)  →  kNN accuracy on held-out scRNA: {acc_before:.4f}")

# ── 6. Fine-tune: close scRNA → spatial domain gap ───────────────────────────
model.finetune_spatial(
    sc_dataset,
    sp_dataset,
    train_epochs=30,
    batch_size=2048,
    early_stopping_patience=10,
    accelerator="gpu",
    devices=1,
)

# ── 7. Re-embed with fine-tuned network & label transfer ──────────────────────
sp_adata.obsm["X_cellpin_match_noft"] = sp_adata.obsm["X_cellpin_match"].copy()
sp_adata.obsm["X_cellpin_match"] = model.embed_atlas(sp_dl)

# Force sc_adata re-embed through fine-tuned weights
if "X_cellpin_match" in sc_adata.obsm:
    del sc_adata.obsm["X_cellpin_match"]

acc_after, sp_adata = model.tl.label_transfer(sc_adata, CELL_TYPE_COL, sp_adata)
print(f"match_emb + finetune      →  kNN accuracy on held-out scRNA: {acc_after:.4f}")

# ── 8. Build co-embedding: scRNA + spatial in shared atlas space ──────────────
sc_adata.obsm["X_coembedding"] = sc_adata.obsm[ATLAS_EMB_KEY].copy()
sp_adata.obsm["X_coembedding"] = sp_adata.obsm["X_cellpin_match"].copy()

sc_adata.obs["plot_label"] = sc_adata.obs[CELL_TYPE_COL].astype(str)
sp_adata.obs["plot_label"] = sp_adata.obs["cellpin_annotation"].astype(str)

combined = ad.concat(
    {"scRNA": sc_adata, "spatial": sp_adata},
    join="inner",
    label="source",
)
print(f"Combined: {combined.n_obs:,} cells  |  X_coembedding: {combined.obsm['X_coembedding'].shape}")

# ── 9. Save ───────────────────────────────────────────────────────────────────
combined.write_h5ad(OUT_PATH)
print(f"Saved → {OUT_PATH}")
