import anndata as ad
import numpy as np
import pytest

from cellpin.dataset import scAnnDataset, stAnnDataset
from cellpin.pp import setup_data


@pytest.fixture
def adata_sc():
    rng = np.random.default_rng(0)
    X = rng.integers(1, 50, size=(10, 20)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene{i}" for i in range(20)]
    adata.layers["counts"] = X.copy()
    return adata


@pytest.fixture
def adata_st():
    rng = np.random.default_rng(1)
    panel_genes = [f"gene{i}" for i in range(8)]
    X = rng.integers(1, 30, size=(5, 8)).astype(np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = panel_genes
    adata.layers["counts"] = X.copy()
    return adata


@pytest.fixture
def sc_dataset(adata_sc, adata_st):
    sc_ds, _ = setup_data(sc_adata=adata_sc, st_adata=adata_st, layer="counts")
    return sc_ds


@pytest.fixture
def st_dataset(adata_sc, adata_st):
    _, st_ds = setup_data(sc_adata=adata_sc, st_adata=adata_st, layer="counts")
    return st_ds
