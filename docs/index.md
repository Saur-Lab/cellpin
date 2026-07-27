# cellpin

```{image} _static/cellpin_fig1.png
:align: center
:alt: cellpin Model Overview
:width: 800px
```

Cellpin is a lightweight probabilistic model that reconstructs and denoises spatial transcriptomes from single-cell RNA-seq references. It enables transcriptome-wide imputation, robust atlas-to-spatial label-transfer, and improved biological interpretation of both targeted-panel and full-transcriptome spatial datasets.

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

## Release notes

See the [changelog](changelog.md).

## Contact

If you found a bug or have a feature request, please use the [issue tracker](https://github.com/Saur-Lab/cellpin/issues).

## Citation

Putze P*, Lucarelli D*, Wellappili D, Bahrami M, Luecken MD, Theis FJ, Saur D. Cellpin enables reference-based imputation and denoising of spatial transcriptomes. bioRxiv 2026.06.02.729566. doi: 10.64898/2026.06.02.729566

```{toctree}
:hidden: true
:maxdepth: 1

api.md
best_practices.md
changelog.md
contributing.md

notebooks/cellpin_tutorial
notebooks/label_transfer
notebooks/atera_whole_transcriptome
```
