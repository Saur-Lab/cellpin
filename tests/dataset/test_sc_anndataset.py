import anndata as ad
import numpy as np
import pytest
import torch
from scipy import sparse

from cellpin.dataset import scAnnDataset


@pytest.fixture
def adata():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    adata = ad.AnnData(X=X)
    adata.var_names = ["gene1", "gene2", "gene3"]
    adata.layers["counts"] = X * 2
    return adata


def test_init(adata):
    ds = scAnnDataset(adata=adata, panel=["gene1", "gene3"])
    assert len(ds) == 2
    assert ds.panel_genes == ["gene1", "gene3"]


def test_getitem_keys(adata):
    ds = scAnnDataset(adata=adata, panel=["gene1", "gene3"])
    item = ds[0]
    assert set(item.keys()) == {"full_expr", "panel_expr", "no_panel_expr", "local_l_mean", "local_l_var"}


def test_getitem_values(adata):
    ds = scAnnDataset(adata=adata, panel=["gene1", "gene3"])
    item = ds[0]
    assert torch.allclose(item["full_expr"], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(item["panel_expr"], torch.tensor([1.0, 3.0]))
    assert torch.allclose(item["no_panel_expr"], torch.tensor([2.0]))


def test_layer(adata):
    ds = scAnnDataset(adata=adata, layer="counts", panel=["gene1"])
    item = ds[0]
    assert torch.allclose(item["full_expr"], torch.tensor([2.0, 4.0, 6.0]))


def test_sparse_input(adata):
    adata.X = sparse.csr_matrix(adata.X)
    ds = scAnnDataset(adata=adata, panel=["gene1", "gene3"])
    item = ds[0]
    assert torch.allclose(item["full_expr"], torch.tensor([1.0, 2.0, 3.0]))


def test_noise_scale_removed(adata):
    import pytest
    with pytest.raises(TypeError, match="noise_scale"):
        scAnnDataset(adata=adata, panel=["gene1"], noise_scale=0.1)


def test_panel_masking(adata):
    ds = scAnnDataset(adata=adata, panel=["gene2"])
    item = ds[0]
    assert torch.allclose(item["panel_expr"], torch.tensor([2.0]))
    assert torch.allclose(item["no_panel_expr"], torch.tensor([1.0, 3.0]))


def test_library_stats(adata):
    ds = scAnnDataset(adata=adata, panel=["gene1"])
    item = ds[0]
    assert item["local_l_mean"].shape == (1,)
    assert item["local_l_var"].shape == (1,)
    assert item["local_l_var"].item() > 0
