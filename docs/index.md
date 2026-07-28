# cellpin

```{image} _static/cellpin_fig1.png
:align: center
:alt: cellpin Model Overview
:width: 800px
```

Cellpin is a lightweight probabilistic model that reconstructs and denoises spatial transcriptomes from single-cell RNA-seq references. It enables transcriptome-wide imputation, robust atlas-to-spatial label-transfer, and improved biological interpretation of both targeted-panel and full-transcriptome spatial datasets.

New here? [Use cases](use_cases.md) covers what cellpin is good for and when to reach for it.

## Installation

Python 3.11 or newer is required.

**pip**

```bash
pip install cellpin
```

**uv**

```bash
uv pip install cellpin
```

For SpatialData support add the spatial extras:

```bash
pip install "cellpin[spatial]"
# or
uv pip install "cellpin[spatial]"
```

## Quickstart

```python
import cellpin
import torch

# sc_adata: annotated scRNA-seq reference   sp_adata: your spatial data
# both need raw integer counts in .X or a named layer
sc_dataset, sp_dataset = cellpin.pp.setup_data(sc_adata, sp_adata, layer="counts")

model = cellpin.CellPin(sc_dataset)
model.fit(sc_dataset)

dl = torch.utils.data.DataLoader(sp_dataset, batch_size=512, shuffle=False)
adata = model.impute(dl, obs_adata=sp_adata, return_norm=True, return_int=True)
```

One forward pass gives you all three outputs at once:

- `adata.obsm["X_cellpin"]`: the cell embedding, ready for `sc.pp.neighbors(use_rep="X_cellpin")`
- `adata.layers["imputed"]`: denoised integer counts across the full reference gene space
- `cellpin.tl.label_transfer(model, sc_adata, "cell_type", adata)`: annotations from the reference

The [basic usage tutorial](notebooks/cellpin_tutorial.ipynb) walks through this end to end, and
[Best Practices](best_practices.md) is worth a read before your first real training run.

## Release notes

See the [changelog](changelog.md).

## Contact

If you found a bug or have a feature request, please use the [issue tracker](https://github.com/Saur-Lab/cellpin/issues).

## Citation

Putze P*, Lucarelli D*, Wellappili D, Bahrami M, Luecken MD, Theis FJ, Saur D. Cellpin enables reference-based imputation and denoising of spatial transcriptomes. bioRxiv 2026.06.02.729566. doi: 10.64898/2026.06.02.729566

```{toctree}
:hidden: true
:maxdepth: 1

use_cases.md
api.md
best_practices.md
changelog.md
contributing.md

notebooks/cellpin_tutorial
notebooks/label_transfer
notebooks/xenium_denoising
notebooks/atera_whole_transcriptome
```
