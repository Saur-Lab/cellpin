"""Tests for cellpin.tl.label_transfer."""

import anndata as ad
import numpy as np
import pytest
from torch.utils.data import DataLoader

from cellpin.dataset import scAnnDataset, stAnnDataset
from cellpin.models import CellPin
from cellpin.pp import setup_data
from cellpin.tl import label_transfer

MINIMAL_CONFIG = {
    "n_latent": 4,
    "n_hidden": 16,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "reconstruction_loss": "nb",
    "kl_warmup_epochs": 0,
}


@pytest.fixture
def data():
    """Raw counts in layers['counts'], log-normalised values in .X (as in most atlases)."""
    rng = np.random.default_rng(0)
    X_sc = rng.integers(1, 50, size=(20, 20)).astype(np.float32)
    sc_adata = ad.AnnData(X=np.log1p(X_sc))
    sc_adata.var_names = [f"gene{i}" for i in range(20)]
    sc_adata.layers["counts"] = X_sc
    sc_adata.obs["cell_type"] = ["A"] * 10 + ["B"] * 10

    X_sp = rng.integers(1, 30, size=(10, 8)).astype(np.float32)
    sp_adata = ad.AnnData(X=np.log1p(X_sp))
    sp_adata.var_names = [f"gene{i}" for i in range(8)]
    sp_adata.layers["counts"] = X_sp

    sc_ds, _ = setup_data(sc_adata, sp_adata, layer="counts")
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()
    return model, sc_adata, sp_adata


def _embedding(model, ds):
    return model.get_cell_embedding(DataLoader(ds, batch_size=8, shuffle=False))


def test_label_transfer_uses_layers(data):
    model, sc_adata, sp_adata = data
    panel = list(model.panel_gene_names)

    acc, out = label_transfer(model, sc_adata, "cell_type", sp_adata, k=3, sc_layer="counts", sp_layer="counts")

    assert 0.0 <= acc <= 1.0
    assert set(out.obs["cellpin_annotation"].unique()) <= {"A", "B"}
    np.testing.assert_allclose(
        sc_adata.obsm["X_cellpin"],
        _embedding(model, scAnnDataset(sc_adata, layer="counts", panel=panel)),
        rtol=1e-4,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sp_adata.obsm["X_cellpin"],
        _embedding(model, stAnnDataset(sp_adata, panel_genes=panel, layer="counts")),
        rtol=1e-4,
        atol=1e-5,
    )


def test_label_transfer_defaults_to_X(data):
    model, sc_adata, sp_adata = data
    panel = list(model.panel_gene_names)

    label_transfer(model, sc_adata, "cell_type", sp_adata, k=3)

    np.testing.assert_allclose(
        sc_adata.obsm["X_cellpin"],
        _embedding(model, scAnnDataset(sc_adata, panel=panel)),
        rtol=1e-4,
        atol=1e-5,
    )


def test_label_transfer_missing_layer_raises(data):
    model, sc_adata, sp_adata = data
    with pytest.raises(ValueError, match="not found in sc_adata.layers"):
        label_transfer(model, sc_adata, "cell_type", sp_adata, sc_layer="nope")
    with pytest.raises(ValueError, match="not found in sp_adata.layers"):
        label_transfer(model, sc_adata, "cell_type", sp_adata, sp_layer="nope")
