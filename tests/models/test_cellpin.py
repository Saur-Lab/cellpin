"""Integration tests for the CellPin Lightning module."""

import anndata as ad
import numpy as np
import pytest
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


def test_cellpin_impute(small_datasets):
    """Basic shape and key checks for impute()."""
    import scipy.sparse as sp

    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)
    adata_out = model.impute(loader, mc_samples=2)

    assert adata_out.n_obs == len(st_ds)
    assert adata_out.n_vars == sc_ds.X.shape[1]
    assert "X_cellpin" in adata_out.obsm
    assert adata_out.obsm["X_cellpin"].shape == (len(st_ds), MINIMAL_CONFIG["n_latent"])
    assert "imputed" in adata_out.layers
    assert "is_measured" in adata_out.var.columns


def test_impute_no_sentinel(small_datasets):
    """No -2 values anywhere in imputed output."""
    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)
    adata_sc_loader = DataLoader(sc_ds, batch_size=4, shuffle=False)

    # impute spatial with obs_adata that has fewer genes (triggers the fill path)
    import anndata as ad
    import numpy as np
    rng = np.random.default_rng(0)
    obs_adata = ad.AnnData(X=rng.integers(1, 10, size=(len(st_ds), 8)).astype(np.float32))
    obs_adata.var_names = [f"gene{i}" for i in range(8)]
    obs_adata.layers["counts"] = obs_adata.X.copy()

    adata_out = model.impute(loader, obs_adata=obs_adata, mc_samples=2, return_sparse=False)

    assert np.all(adata_out.X >= 0), "X contains negative values"
    for lyr in adata_out.layers.values():
        arr = lyr.toarray() if hasattr(lyr, "toarray") else lyr
        assert np.all(arr >= 0), f"Layer contains negative values"


def test_is_measured_var_column(small_datasets):
    """is_measured correctly marks panel genes vs unmeasured genes."""
    import anndata as ad
    import numpy as np

    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)

    # obs_adata has only 8 of the 20 genes
    rng = np.random.default_rng(1)
    obs_adata = ad.AnnData(X=rng.integers(1, 10, size=(len(st_ds), 8)).astype(np.float32))
    obs_adata.var_names = [f"gene{i}" for i in range(8)]
    obs_adata.layers["counts"] = obs_adata.X.copy()

    adata_out = model.impute(loader, obs_adata=obs_adata, mc_samples=2)

    assert "is_measured" in adata_out.var.columns
    assert adata_out.var["is_measured"].dtype == bool
    # genes 0-7 are in obs_adata, genes 8-19 are not
    assert adata_out.var.loc["gene0", "is_measured"] is np.bool_(True)
    assert adata_out.var.loc["gene7", "is_measured"] is np.bool_(True)
    assert adata_out.var.loc["gene8", "is_measured"] is np.bool_(False)
    assert adata_out.var.loc["gene19", "is_measured"] is np.bool_(False)
    assert adata_out.var["is_measured"].sum() == 8


def test_impute_sparse_output(small_datasets):
    """X and layers are sparse by default; dense when return_sparse=False."""
    import scipy.sparse as sp

    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)

    adata_sparse = model.impute(loader, mc_samples=2, return_sparse=True)
    assert sp.issparse(adata_sparse.X)
    assert sp.issparse(adata_sparse.layers["imputed"])

    adata_dense = model.impute(loader, mc_samples=2, return_sparse=False)
    assert not sp.issparse(adata_dense.X)
    assert not sp.issparse(adata_dense.layers["imputed"])


def test_impute_return_int_sparse(small_datasets):
    """return_int=True produces sparse int32 with no negative values."""
    import scipy.sparse as sp

    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)
    adata_out = model.impute(loader, mc_samples=2, return_int=True, return_sparse=True)

    assert sp.issparse(adata_out.X)
    assert adata_out.X.dtype == np.int32
    assert sp.issparse(adata_out.layers["imputed"])
    assert adata_out.layers["imputed"].dtype == np.int32
    assert adata_out.X.min() >= 0
