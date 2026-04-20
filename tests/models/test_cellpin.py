"""Integration tests for the CellPin Lightning module."""

import anndata as ad
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cellpin.models import CellPin
from cellpin.pp import setup_data


MINIMAL_CONFIG = {
    "n_latent": 4,
    "n_hidden": 16,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "reconstruction_loss": "nb",
    "kl_warmup_epochs": 0,
}

TRAINER_KWARGS = {
    "max_epochs": 1,
    "accelerator": "cpu",
    "devices": 1,
    "enable_early_stopping": False,
    "enable_checkpointing": False,
    "enable_progress_bar": False,
    "batch_size": 4,
    "log_every_n_steps": 1,
}


@pytest.fixture
def small_datasets():
    rng = np.random.default_rng(42)
    # sc: 20 cells × 20 genes
    X_sc = rng.integers(1, 50, size=(20, 20)).astype(np.float32)
    adata_sc = ad.AnnData(X=X_sc)
    adata_sc.var_names = [f"gene{i}" for i in range(20)]
    adata_sc.layers["counts"] = X_sc.copy()

    # st: 10 cells × 8 genes (subset overlap)
    X_st = rng.integers(1, 30, size=(10, 8)).astype(np.float32)
    adata_st = ad.AnnData(X=X_st)
    adata_st.var_names = [f"gene{i}" for i in range(8)]
    adata_st.layers["counts"] = X_st.copy()

    sc_ds, st_ds = setup_data(sc_adata=adata_sc, st_adata=adata_st, layer="counts")
    return sc_ds, st_ds


def test_cellpin_train_model(small_datasets, tmp_path):
    sc_ds, _ = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)

    model.train_model(
        dataset=sc_ds,
        freeze_pretrained=False,
        require_pretrained=False,
        output_dir=str(tmp_path / "run"),
        **TRAINER_KWARGS,
    )


def test_cellpin_get_cell_embedding(small_datasets):
    sc_ds, _ = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(sc_ds, batch_size=4, shuffle=False)
    embeddings = model.get_cell_embedding(loader)

    assert embeddings.shape == (len(sc_ds), MINIMAL_CONFIG["n_latent"])
    assert embeddings.dtype == np.float32


def test_cellpin_impute_to_anndata(small_datasets):
    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)
    adata_out = model.impute_to_anndata(loader)

    assert adata_out.n_obs == len(st_ds)
    assert adata_out.n_vars == sc_ds.X.shape[1]
    assert "X_cellpin" in adata_out.obsm
    assert adata_out.obsm["X_cellpin"].shape == (len(st_ds), MINIMAL_CONFIG["n_latent"])
    assert "imputed" in adata_out.layers
