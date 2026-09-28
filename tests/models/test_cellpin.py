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


def test_impute_return_norm(small_datasets):
    """return_norm=True adds a finite, non-negative imputed_norm layer."""
    import scipy.sparse as sp

    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)
    adata_out = model.impute(loader, mc_samples=2, return_norm=True, nb_count_samples=10, return_sparse=False)

    norm = adata_out.layers["imputed_norm"]
    assert not sp.issparse(norm)
    assert norm.shape == (len(st_ds), sc_ds.X.shape[1])
    assert norm.dtype == np.float32
    assert np.isfinite(norm).all()
    assert (norm >= 0).all()

    adata_sparse = model.impute(loader, mc_samples=2, return_norm=True, nb_count_samples=10, return_sparse=True)
    assert sp.issparse(adata_sparse.layers["imputed_norm"])


def test_impute_nb_seed_is_reproducible(small_datasets):
    """nb_seed pins the imputed_norm layer; omitting it does not."""
    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)
    kwargs = {
        "mc_samples": 2,
        "return_norm": True,
        "nb_count_samples": 10,
        "return_sparse": False,
    }
    # mc_impute makes the counts themselves stochastic, so pin the whole run.
    torch.manual_seed(0)
    a = model.impute(loader, nb_seed=123, **kwargs).layers["imputed_norm"]
    torch.manual_seed(0)
    b = model.impute(loader, nb_seed=123, **kwargs).layers["imputed_norm"]
    torch.manual_seed(0)
    c = model.impute(loader, nb_seed=456, **kwargs).layers["imputed_norm"]

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_impute_return_norm_area_key(small_datasets):
    """area_key switches to area normalisation and rejects non-positive areas."""
    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()

    loader = DataLoader(st_ds, batch_size=4, shuffle=False)
    obs_adata = ad.AnnData(X=np.zeros((len(st_ds), 0), dtype=np.float32))
    obs_adata.obs["cell_area"] = np.linspace(20.0, 100.0, len(st_ds))

    adata_out = model.impute(
        loader,
        obs_adata=obs_adata,
        mc_samples=2,
        return_norm=True,
        nb_count_samples=10,
        area_key="cell_area",
        return_sparse=False,
    )
    assert np.isfinite(adata_out.layers["imputed_norm"]).all()

    with pytest.raises(ValueError, match="not found in adata.obs"):
        model.impute(loader, obs_adata=obs_adata, mc_samples=2, return_norm=True, area_key="missing")

    obs_adata.obs["cell_area"] = 0.0
    with pytest.raises(ValueError, match="areas must be positive"):
        model.impute(loader, obs_adata=obs_adata, mc_samples=2, return_norm=True, area_key="cell_area")


def test_impute_return_var(small_datasets):
    """return_var adds a non-negative variance layer; deterministic modes give zero variance."""
    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    model.eval()
    loader = DataLoader(st_ds, batch_size=4, shuffle=False)

    adata_out = model.impute(loader, mc_samples=4, return_var=True, return_sparse=False)
    var = adata_out.layers["imputed_var"]
    assert var.shape == (len(st_ds), sc_ds.X.shape[1])
    assert np.isfinite(var).all() and (var >= 0).all()
    assert var.max() > 0

    det = model.impute(loader, mc_impute=False, return_var=True, return_sparse=False)
    assert np.all(det.layers["imputed_var"] == 0)

    # posterior-mean latent and no masking: every MC pass is identical
    no_noise = model.impute(
        loader, mc_samples=3, mask_fraction=0.0, mc_latent_sample=False, return_var=True, return_sparse=False
    )
    np.testing.assert_allclose(no_noise.layers["imputed_var"], 0.0, atol=1e-6)
    np.testing.assert_allclose(no_noise.layers["imputed"], det.layers["imputed"], rtol=1e-5, atol=1e-6)


FIT_KWARGS = {
    "pretrain_epochs": 1,
    "train_epochs": 1,
    "batch_size": 4,
    "accelerator": "cpu",
    "devices": 1,
    "enable_progress_bar": False,
    "log_every_n_steps": 1,
}


def _params(module):
    # trainable parameters only (BatchNorm running stats update in train mode even when frozen)
    return {k: v.detach().clone() for k, v in module.named_parameters()}


def _changed(before, module):
    after = _params(module)
    return any(not torch.equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize(
    ("overrides", "fit_kwargs", "full_changes", "panel_changes", "decoder_changes"),
    [
        ({}, {}, True, True, True),  # two-stage default
        ({}, {"freeze_pretrained": True}, True, True, True),  # decoder frozen only in stage 2
        ({"lambda_inv": 0.0}, {}, True, True, True),
        ({"training_mode": "plain_ae"}, {}, False, True, True),
        ({"training_mode": "single_stage"}, {}, True, True, True),
        ({"library_mode": "observed_panel"}, {}, True, True, True),
    ],
)
def test_fit_ablation_modes(
    small_datasets, tmp_path, overrides, fit_kwargs, full_changes, panel_changes, decoder_changes
):
    """Each ablation switch trains end to end and updates the expected modules."""
    sc_ds, st_ds = small_datasets
    model = CellPin(sc_dataset=sc_ds, config={**MINIMAL_CONFIG, **overrides})
    before = {m: _params(getattr(model.vae, m)) for m in ("z_encoder_full", "z_encoder_panel", "decoder", "l_encoder")}

    model.fit(sc_ds, output_dir=str(tmp_path / "run"), **FIT_KWARGS, **fit_kwargs)

    assert _changed(before["z_encoder_full"], model.vae.z_encoder_full) == full_changes
    assert _changed(before["z_encoder_panel"], model.vae.z_encoder_panel) == panel_changes
    assert _changed(before["decoder"], model.vae.decoder) == decoder_changes
    assert _changed(before["l_encoder"], model.vae.l_encoder) == (overrides.get("library_mode") != "observed_panel")

    out = model.impute(DataLoader(st_ds, batch_size=4), mc_samples=2, return_sparse=False)
    assert np.isfinite(out.layers["imputed"]).all()


def test_decoder_frozen_stage2(small_datasets, tmp_path):
    """freeze_pretrained=True keeps the decoder fixed during stage 2."""
    sc_ds, _ = small_datasets
    model = CellPin(sc_dataset=sc_ds, config=MINIMAL_CONFIG)
    kw = {k: v for k, v in FIT_KWARGS.items() if k not in ("pretrain_epochs", "train_epochs")}
    model.pretrain_model(sc_ds, pretrain_epochs=1, output_dir=str(tmp_path / "pre"), **kw)
    before = _params(model.vae.decoder)
    model.train_model(sc_ds, freeze_pretrained=True, max_epochs=1, output_dir=str(tmp_path / "train"), **kw)
    assert not _changed(before, model.vae.decoder)


def test_observed_panel_library(small_datasets):
    """library_mode='observed_panel' uses log panel totals and no library KL."""
    sc_ds, _ = small_datasets
    model = CellPin(sc_dataset=sc_ds, config={**MINIMAL_CONFIG, "library_mode": "observed_panel"})
    batch = next(iter(DataLoader(sc_ds, batch_size=5)))
    x_panel = batch["panel_expr"]
    ql_m, ql_v, lib = model.vae.infer_library(x_panel, torch.log1p(x_panel))
    torch.testing.assert_close(lib, torch.log(x_panel.sum(1, keepdim=True)))
    assert torch.all(ql_v == 0)
    model.train()
    assert torch.all(model.compute_losses(batch)["kl_l_loss"] == 0)


def test_no_inv_skips_teacher(small_datasets):
    """With lambda_inv=0 the invariance terms are zero and the teacher is not run."""
    sc_ds, _ = small_datasets
    model = CellPin(sc_dataset=sc_ds, config={**MINIMAL_CONFIG, "lambda_inv": 0.0})
    calls = []
    orig = model.vae.inference
    model.vae.inference = lambda *a, **k: calls.append(k.get("encoder_view")) or orig(*a, **k)
    losses = model.compute_losses(next(iter(DataLoader(sc_ds, batch_size=5))))
    assert calls == ["panel"]
    assert losses["distill_loss"] == 0 and losses["snn_loss"] == 0


def test_invalid_training_mode(small_datasets):
    sc_ds, _ = small_datasets
    with pytest.raises(ValueError, match="training_mode"):
        CellPin(sc_dataset=sc_ds, config={**MINIMAL_CONFIG, "training_mode": "bogus"})


def test_student_decoder_loss_switches(small_datasets):
    """lambda_recon=0 + lambda_pearson_train=0: Stage 2 sends no gradient to the decoder; Stage 1 keeps Pearson."""
    sc_ds, _ = small_datasets
    cfg = {**MINIMAL_CONFIG, "lambda_pearson": 0.5, "lambda_pearson_train": 0.0, "lambda_recon": 0.0}
    model = CellPin(sc_dataset=sc_ds, config=cfg)
    model.train()
    batch = next(iter(DataLoader(sc_ds, batch_size=5)))
    assert model.compute_pretrain_losses(batch)["pearson_loss"] > 0
    losses = model.compute_losses(batch)
    assert losses["pearson_loss"] == 0
    model.zero_grad()
    losses["loss"].backward()
    assert all(p.grad is None or torch.all(p.grad == 0) for p in model.vae.decoder.parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.vae.z_encoder_panel.parameters())
