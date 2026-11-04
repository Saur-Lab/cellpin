---
name: cellpin
description: Run cellpin on single-cell-resolved spatial transcriptomics (Xenium, CosMx, MERSCOPE, Atera, ...) with an scRNA-seq reference to get cell embeddings, imputed/denoised expression, and label transfer. Use whenever the user mentions cellpin, wants to impute or denoise spatial genes from a single-cell reference, embed spatial cells with a reference, or transfer cell-type labels from an atlas to spatial data.
---

# cellpin

cellpin trains on an scRNA-seq **reference** and then encodes **spatial** cells from their measured
panel genes. One trained model plus one forward pass (`model.impute`) gives three outputs:

| Output | Where it lands | Use it for |
|---|---|---|
| Embedding | `adata.obsm["X_cellpin"]` | neighbors / Leiden / UMAP (replaces PCA) |
| Imputed + denoised expression | `adata.layers["imputed"]`, `adata.layers["imputed_norm"]` | unmeasured genes, denoised measured genes |
| Labels | `cellpin.tl.label_transfer(...)` → `obs["cellpin_annotation"]` | annotation from the reference |

Docs: https://cellpin.readthedocs.io · Paper: https://doi.org/10.64898/2026.06.02.729566

## Follow these steps in order. Do not skip any.

### Step 1: Intake (ask the user before you write any code)

Ask for everything below in **one message**. Infer what you can by inspecting the files first, and then ask the user only to confirm.

1. **Goal**: embedding only / imputation / denoising / label transfer (more than one is fine).
2. **Spatial data**: path, platform (Xenium, CosMx, MERSCOPE, Atera, other), format (`.h5ad`, SpatialData `.zarr`, or a raw output folder).
3. **Reference data**: path, tissue, species. It must match the tissue and species of the spatial data.
4. **Raw counts location** for both objects (`.X`, `.raw.X`, or a layer such as `"counts"`).
5. **Batch column** in the reference (study, donor or sample). This is strongly recommended for atlases.
6. **Label column** in the reference (only for label transfer). Prefer a coarse level first.
7. **Output format**: Jupyter notebook or Python script.
8. **Hardware**: is a GPU available? CPU works but is slow.
9. **Output directory** for results.

If the user has no data and just wants to try cellpin, use the example data
(`cellpin.pp.load_sc_example()`, `cellpin.pp.load_sp_example()`, `LAYER="counts"`), which is the
setup from the basic tutorial.

### Step 2: Preflight check (always run it, and fix every FAIL)

Run the bundled checker. The script lives next to this file in `scripts/check_inputs.py`:

```bash
python <this-skill-dir>/scripts/check_inputs.py \
    --sc reference.h5ad --sp spatial.h5ad \
    --layer counts --batch-key batch --label-key cell_type
```

Run it **after** any preprocessing (loading counts into a layer, gene subsetting) on the files or
objects you will actually pass to cellpin. If it is easier to do this in Python, the checks are
also available as a function:
`from check_inputs import run_checks; run_checks(sc_adata, sp_adata, layer=..., batch_key=..., label_key=...)`.

- **FAIL** → stop and fix it (`references/troubleshooting.md` covers each one). Do not train.
- **WARN** → tell the user, fix it if the fix is cheap, and otherwise continue.

### Step 3: Prepare the data

Follow `references/workflow.md` → "Prepare". In short:

- Put the raw integer counts **under the same layer name in both objects** (for example `layers["counts"]`).
  `setup_data` uses a single `layer=` for both objects.
- Keep a measured, normalized layer on the spatial data (`layers["log1p_norm"]`) so you can compare
  measured and imputed values later.
- Apply light QC to the spatial data (drop cells with fewer than ~10 genes for targeted panels; see Atera for WT).
- **Gene space of the reference**:
  - Targeted panels (Xenium/CosMx/MERSCOPE): **all panel genes + 1,000–2,000 batch-aware HVGs**.
  - Whole transcriptome (Atera etc.): **~4,000 batch-aware reference HVGs, restricted to genes the spatial data measured**.
  - Never pass the full ~30k-gene reference.
- Make gene identifiers match (both symbols or both Ensembl IDs) and make them unique.

### Step 4: Train

```python
sc_dataset, sp_dataset = cellpin.pp.setup_data(sc_adata, sp_adata, layer=LAYER, batch_key=BATCH_KEY)
model = cellpin.CellPin(sc_dataset)
model.fit(sc_dataset, pretrain_epochs=50, train_epochs=60, batch_size=BATCH_SIZE)
model.pl.losses()
```

- `BATCH_SIZE`: **if the reference has more than ~30,000 cells, use `batch_size=512`**. This is strongly
  recommended, because larger batches give the SNN loss more diversity and stabilize training. Use 1024 for very
  large atlases. Use 256 (the default) only for references with 30k cells or fewer.
- **Do not turn on checkpointing** (`save_checkpoints=False` is the default). The model you get after `fit` is the
  last epoch, which is what you want, and training is much faster without checkpointing. Only if the loss curves show
  clear divergence should you retrain with `save_checkpoints=True`. `fit` then automatically restores the best (earlier)
  checkpoint (see `references/workflow.md` → "Checkpoints").
- **Keep the default hyperparameters** for the first run. If the results are unsatisfying after validation (Step 7),
  tuning is encouraged: learning rate, epochs, loss weights, `distillation_mode`, or the KL floor below.
  Change one thing at a time and compare against the first run.
- **Sharper reconstructions (optional, never the default)**: if cells of the same type look over-smoothed, set the KL
  floor `kl_free_bits=2.0` **and** switch early stopping to monitor the reconstruction loss:
  ```python
  model = cellpin.CellPin(sc_dataset, config={"kl_free_bits": 2.0})
  model.fit(sc_dataset, batch_size=BATCH_SIZE, checkpoint_monitor="val_reconst_loss")
  ```
  With the floor, the total `val_loss` contains a KL term that cannot drop below the floor. Early stopping on
  `val_loss` then stops too early or at the wrong point, so monitor `val_reconst_loss` instead.
- Read the `setup_data` printout. The panel overlap should cover most spatial genes.
- `pl.losses()` shows **Stage 2 only**. Flat or slightly jagged curves are normal. Only `NaN`, divergence, or a steadily climbing loss are real problems.

### Step 5: Embed / impute

```python
loader = torch.utils.data.DataLoader(sp_dataset, batch_size=1024, shuffle=False)  # shuffle=False is mandatory
adata = model.impute(
    loader, obs_adata=sp_adata, return_norm=True, return_int=True, area_key="cell_area", nb_count_samples=20
)
```

- Use `return_int=True` for any downstream analysis, because it gives a real denoised count matrix.
  (The tutorials leave it off only for correlation benchmarks.)
- Pass `area_key="cell_area"` if that column exists (it does for Xenium and Atera). Otherwise pass `area_key=None`.
- **Embedding or labels only** (no expression needed): use the fast path and add
  `mc_samples=1, mask_fraction=0.0`.
- `obs_adata` must be the exact object that `sp_dataset` was built from, so that cells stay in the same order.

### Step 6: Label transfer (only if requested)

```python
acc, adata = cellpin.tl.label_transfer(
    model, sc_adata, LABEL_KEY, adata, conf_threshold=0.5, sc_layer=LAYER, sp_layer=LAYER
)
```

**Always pass `sc_layer=LAYER` (and `sp_layer=LAYER`) when the raw counts are in a layer.** If you leave them out,
`label_transfer` embeds the cells from `.X`, and most atlases keep log-normalized values in `.X`, which gives a
wrong reference embedding. (The `adata` returned by `impute` already has `X_cellpin`, so `sp_layer` only matters if
you pass a different object.) `sc_layer`/`sp_layer` need cellpin > 0.3.0. On older versions, set
`sc_adata.obsm["X_cellpin"] = model.get_cell_embedding(DataLoader(sc_dataset, batch_size=1024, shuffle=False))` first.

The printed accuracy is measured on held-out **reference** cells. It does not show that the labels are correct in the tissue.

### Step 7: Validate (required, never skip)

- Build neighbors on the embedding: `sc.pp.neighbors(adata, use_rep="X_cellpin")`, then UMAP and Leiden.
- Plot 3–6 known markers in **measured** (`log1p_norm`) and **imputed** (`imputed_norm`) layers,
  side by side on UMAP and in space. Use only genes in the panel for the measured layer
  (`adata.var["is_measured"]`). Unmeasured genes show as 0 in measured layers.
- For label transfer, check that each label's markers light up in the measured counts.
- Report to the user what looks good and what looks off. Do not claim success without these checks.

### Step 8: Save

`adata.write_h5ad(OUT_DIR / "cellpin_output.h5ad")`. Save the model with `model.save(OUT_DIR / "cellpin_model.pt")`
if the user may want it later. Also record the commands or parameters used.

## Golden rules

1. **Raw integer counts only** as model input. Log-normalized or scaled data silently produces bad results.
2. **The same layer name holds raw counts in both objects.**
3. **Gene identifiers must match** (symbols vs Ensembl, case, species).
4. **Use 20,000 or more reference cells.** The reference must come from the same tissue and species. The better the reference, the better the result. 10x Flex references work especially well.
5. **Pass `batch_key`** whenever the reference combines several studies, donors or batches.
6. **Use default hyperparameters first.** If the results are unsatisfying, tuning is encouraged (see `references/workflow.md` → "Tuning").
7. **More than 30k reference cells → `batch_size=512`.**
8. **No checkpointing by default.** Use the last epoch. Checkpoints are only for runs that clearly diverge.
9. **Always use `shuffle=False`** in DataLoaders used for imputation or embedding.
10. **Imputed values are model estimates, not measurements.** Do not run formal DE, cell–cell communication, or draw headline conclusions from imputed-only genes without checking them against measured genes or independent data. Say this to the user whenever they head in that direction.
11. **Check labels against markers.** Transferred labels are hypotheses until markers in the measured counts confirm them.
12. Segmentation quality still matters. cellpin denoises, but it does not segment. Cellpose or 10x multimodal segmentation works best.

## Reference files (read when needed)

- `references/workflow.md`: complete copy-paste code for every step, plus platform variants
  (AnnData, SpatialData, Atera whole transcriptome, CELLxGENE atlases, held-out evaluation).
- `references/troubleshooting.md`: preflight FAIL/WARN fixes, error messages, poor results.
- `scripts/check_inputs.py`: the preflight checker.

Tutorials (for more depth):
[basic usage](https://cellpin.readthedocs.io/en/latest/notebooks/cellpin_tutorial.html) ·
[label transfer](https://cellpin.readthedocs.io/en/latest/notebooks/label_transfer.html) ·
[Xenium denoising](https://cellpin.readthedocs.io/en/latest/notebooks/xenium_denoising.html) ·
[Atera whole transcriptome](https://cellpin.readthedocs.io/en/latest/notebooks/atera_whole_transcriptome.html) ·
[best practices](https://cellpin.readthedocs.io/en/latest/best_practices.html)
