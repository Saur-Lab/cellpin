# cellpin

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

Cellpin is a lightweight probabilistic model that reconstructs and denoises spatial transcriptomes from single-cell RNA-seq references. It enables transcriptome-wide imputation, robust atlas-to-spatial label-transfer, and improved biological interpretation of both targeted-panel and full-transcriptome spatial datasets.

## Installation

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

## Documentation

Full documentation, tutorials, and API reference: [cellpin.readthedocs.io](https://cellpin.readthedocs.io/)
