import anndata as ad
import numpy as np
import pytest

from cellpin.dataset import scAnnDataset, stAnnDataset
from cellpin.pp import setup_data


@pytest.fixture
def sc_adata():
    X = np.arange(1, 31, dtype=np.float32).reshape(2, 15)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene{i}" for i in range(15)]
    return adata


@pytest.fixture
def st_adata():
    X = np.ones((3, 5), dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"gene{i}" for i in range(5)]  # gene0..gene4 overlap with sc
    return adata


def test_setup_data_returns_correct_types(sc_adata, st_adata):
    sc_ds, st_ds = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    assert isinstance(sc_ds, scAnnDataset)
    assert isinstance(st_ds, stAnnDataset)


def test_panel_genes_match(sc_adata, st_adata):
    sc_ds, st_ds = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    assert sc_ds.panel_genes == st_ds.panel_genes


def test_panel_genes_are_spatial_genes(sc_adata, st_adata):
    sc_ds, st_ds = setup_data(sc_adata=sc_adata, st_adata=st_adata)
    spatial_genes = set(st_adata.var_names.tolist())
    assert set(sc_ds.panel_genes).issubset(spatial_genes)


def test_no_overlap_raises():
    sc = ad.AnnData(X=np.ones((2, 3), dtype=np.float32))
    sc.var_names = ["a", "b", "c"]
    st = ad.AnnData(X=np.ones((2, 2), dtype=np.float32))
    st.var_names = ["x", "y"]
    with pytest.raises(ValueError, match="No overlapping genes"):
        setup_data(sc_adata=sc, st_adata=st)
