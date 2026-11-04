# Troubleshooting

## Preflight FAIL / WARN → fix

| Check | Meaning | Fix |
|---|---|---|
| `counts not raw integers` | The layer holds normalized, log or scaled values | Find the raw counts: `.raw.X`, `layers["counts"]` / `["count"]` / `["raw"]`. The checker lists candidates. Never round normalized data to fake counts. |
| `negative values` / `NaN` | Scaled or corrupted matrix | Same as above: use the raw counts. |
| `layer missing` | `LAYER` does not exist in one object | Copy the raw counts into `layers[LAYER]` in **both** objects (setup uses one layer name). |
| `low gene overlap` | Identifiers don't match | See "Gene identifier mismatch" below. |
| `duplicate gene names` | `setup_data` will raise | `adata.var_names_make_unique()`. Do this after any renaming. |
| `reference < 20,000 cells` | Too few cells for reliable imputation | Use a larger or public atlas of the same tissue. If none exists, continue with a warning to the user. |
| `reference has N genes` (large) | The full transcriptome was passed | Subset to panel + 1,000–2,000 HVGs (targeted) or ~4,000 HVGs (WT). See workflow §3d. |
| `batch_key missing` | Typo, or the column is absent | `print(sc_adata.obs.columns)` and pick the study, donor or sample column. |
| `label class with < 2 cells` | `label_transfer` crashes on a stratified split | Drop or merge rare classes, or use a coarser label column. |
| `area_key missing` | No `cell_area` in spatial obs | Pass `area_key=None` to `impute` (it falls back to total-count normalization). |
| `cells with < 10 genes` (many) | Segmentation debris | `sc.pp.filter_cells(sp_adata, min_genes=10)`. |
| `no GPU` | CUDA not available | It works on CPU but is slow. Subsample the reference and warn the user about the runtime. |

## Gene identifier mismatch (near-zero overlap)

- **Ensembl IDs vs symbols** (`ENSG...` vs `CD3E`). This is very common with CELLxGENE atlases. Map the IDs with the
  symbol column: `sc_adata.var_names = sc_adata.var["feature_name"].astype(str); sc_adata.var_names_make_unique()`.
  The checker prints which `var` column would give the best overlap.
- **Case** (`Cd3e` vs `CD3E`): this usually means mouse and human are mixed. Check the species. Do **not**
  just uppercase mouse genes to match a human reference. Use a reference of the same species, or a proper
  ortholog mapping, and tell the user about the limitation.
- **Version suffixes** (`ENSG000001.12`): strip them with `.str.split(".").str[0]` in both objects.
- **Control probes** (`NegControlProbe_*`, `BLANK_*`, `Unassigned*`): drop them from the spatial data. They
  are not in the reference anyway.

## Errors

| Error | Cause | Fix |
|---|---|---|
| `ValueError: ... duplicate gene name(s)` | Non-unique `var_names` | `var_names_make_unique()` |
| `ValueError: No overlapping genes` | Identifier mismatch | See above |
| `obs_adata has N cells; imputation has M` | `obs_adata` is not the object the dataset was built from (filtered afterwards) | Pass the exact `sp_adata` used in `setup_data`, or rerun `setup_data` after filtering |
| `area_key ... not found` / `cell area <= 0` | Missing or zero areas | `area_key=None`, or drop cells with area ≤ 0 |
| `Column 'X' not found in sc_adata.obs` (label_transfer) | Wrong `LABEL_KEY` | Check `sc_adata.obs.columns` |
| `Layer 'X' not found in sc_adata.layers` (label_transfer) | Wrong `sc_layer`/`sp_layer` | Pass the layer that holds the raw counts |
| `label_transfer() got an unexpected keyword argument 'sc_layer'` | cellpin ≤ 0.3.0 | Upgrade, or set `sc_adata.obsm["X_cellpin"] = model.get_cell_embedding(DataLoader(sc_dataset, shuffle=False))` first |
| `The least populated class in y has only 1 member` | Rare label | Drop or merge classes with fewer than 2 cells |
| CUDA out of memory | Batch too large or data too large | Lower the `batch_size` in `fit` or the DataLoader, and drop unneeded layers before `impute` |
| Host RAM explodes in `impute` | Many layers × many cells × reference genes, all dense | Delete extra layers from `sp_adata` before `impute`, and keep the reference gene space at ~2–4k |
| `KeyError` on a layer | `LAYER` missing in one object | Add it to both |

## Results look bad

1. **Rerun the preflight.** Most bad results come from normalized input or poor gene overlap.
2. **Check the reference.** Is it the same tissue and species? Does it contain the cell types present in the tissue? Does it have 20k or more cells?
   Are its labels any good? A poor reference is the most common cause of poor results.
3. **Batch effects in the embedding** → set `batch_key`.
4. **Batch size**: with more than 30k reference cells, use `batch_size=512` (or 1024 for very large atlases).
5. **Loss curve**: only `NaN`, divergence, or a steadily climbing loss are real problems. A flat Stage 2 is normal.
   For real divergence, retrain with `save_checkpoints=True` and load the best (earlier) checkpoint (workflow.md → "Checkpoints").
6. **Over-smoothed expression** → `config={"kl_free_bits": 2.0}` together with `fit(..., checkpoint_monitor="val_reconst_loss")`.
7. **Still not satisfied** → tune (workflow.md → "Tuning"): `lr`, epochs, loss weights, `distillation_mode`. Change one thing at a time.
8. **Labels disagree with markers** → use a coarser label column, set `conf_threshold=0.5` or higher, and check that the
   reference contains that cell type. Remember that the reference accuracy says nothing about the labels in the tissue.
9. **Segmentation**: heavily mis-segmented cells limit everything. Suggest Cellpose or 10x multimodal segmentation.

## Interpretation guardrails (tell the user)

- Imputed genes are **estimates**. They are fine for exploration, visualization, hypothesis generation, and finding cell types whose
  markers are missing from the panel. They are not ground truth.
- Before formal DE, cell–cell communication analysis, or any conclusion that rests mainly on imputed genes, cross-check
  against measured genes or independent data. Treat the imputed results as supporting evidence only.
- The embedding is often the most valuable output. Use it for clustering and neighborhoods even when imputation isn't needed.
