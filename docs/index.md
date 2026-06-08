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

> t.b.a

```{toctree}
:hidden: true
:maxdepth: 1

api.md
changelog.md
contributing.md

notebooks/cellpin_tutorial
notebooks/label_transfer
```
