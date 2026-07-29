"""Tests for the Monte-Carlo NB normalisation helper."""

import numpy as np
import pytest
import torch

from cellpin.models.utils import mc_log1p_norm
from cellpin.models.utils.nb_sampling import _cell_chunks, _chunk_torch

TARGET = 1e3


@pytest.fixture
def mu_theta():
    rng = np.random.default_rng(0)
    mu = rng.gamma(0.4, 2.5, size=(120, 60)).astype(np.float32)
    theta = np.exp(rng.normal(size=60))
    return mu, theta


def _reference(mu, theta, scale, n_samples, seed):
    """Original impute() implementation, kept as an independent reference."""
    rng = np.random.default_rng(seed)
    mu64 = mu.astype(np.float64)
    p = theta / (theta + np.clip(mu64, 1e-8, None))
    acc = np.zeros_like(mu64)
    for _ in range(n_samples):
        draw = rng.negative_binomial(theta, p).astype(np.float64)
        if scale is not None:
            normed = draw * scale[:, np.newaxis]
        else:
            normed = draw * (TARGET / draw.sum(axis=1, keepdims=True).clip(1e-12))
        acc += np.log1p(normed)
    return (acc / n_samples).astype(np.float32)


def test_shape_and_dtype(mu_theta):
    mu, theta = mu_theta
    out = mc_log1p_norm(mu, theta, 5, TARGET, seed=0)
    assert out.shape == mu.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_seed_is_reproducible_across_thread_counts(mu_theta):
    mu, theta = mu_theta
    a = mc_log1p_norm(mu, theta, 5, TARGET, seed=7, n_threads=1)
    b = mc_log1p_norm(mu, theta, 5, TARGET, seed=7, n_threads=8)
    c = mc_log1p_norm(mu, theta, 5, TARGET, seed=8, n_threads=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_unseeded_runs_differ(mu_theta):
    mu, theta = mu_theta
    a = mc_log1p_norm(mu, theta, 5, TARGET)
    b = mc_log1p_norm(mu, theta, 5, TARGET)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("use_scale", [True, False])
def test_matches_reference_within_mc_error(mu_theta, use_scale):
    """New estimator must sit within the reference's own run-to-run spread."""
    mu, theta = mu_theta
    scale = np.full(mu.shape[0], 4.0) if use_scale else None
    k = 400
    ref_a = _reference(mu, theta, scale, k, seed=1)
    ref_b = _reference(mu, theta, scale, k, seed=2)
    new = mc_log1p_norm(mu, theta, k, TARGET, scale=scale, seed=3)

    baseline = np.sqrt(((ref_a - ref_b) ** 2).mean())
    cross = np.sqrt(((new - ref_a) ** 2).mean())
    assert cross < baseline * 1.2
    assert abs(float(np.mean(new - ref_a))) < 0.1 * baseline


@pytest.mark.parametrize("use_scale", [True, False])
def test_torch_backend_matches_reference(mu_theta, use_scale):
    """Gamma-Poisson backend targets the same expectation as the NB sampler."""
    mu, theta = mu_theta
    scale = np.full(mu.shape[0], 4.0, dtype=np.float32) if use_scale else None
    k = 400
    ref_a = _reference(mu, theta, None if scale is None else scale.astype(np.float64), k, seed=1)
    ref_b = _reference(mu, theta, None if scale is None else scale.astype(np.float64), k, seed=2)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)
    new = _chunk_torch(
        np.ascontiguousarray(mu, np.float32),
        torch.from_numpy(theta).float(),
        scale,
        k,
        TARGET,
        generator,
        torch.device("cpu"),
    )

    baseline = np.sqrt(((ref_a - ref_b) ** 2).mean())
    assert np.sqrt(((new - ref_a) ** 2).mean()) < baseline * 1.2


def test_jensen_correction_is_downward(mu_theta):
    """Sampling inside log1p must sit below the plug-in log1p(norm(E[X]))."""
    mu, theta = mu_theta
    scale = np.full(mu.shape[0], 4.0)
    corrected = mc_log1p_norm(mu, theta, 400, TARGET, scale=scale, seed=0)
    plugin = np.log1p(mu.astype(np.float64) * scale[:, None])
    assert corrected.mean() < plugin.mean()


def test_chunks_tile_the_cell_axis():
    for n_cells, n_genes in ((1, 10), (7, 10), (5000, 2000), (200_000, 5000)):
        chunks = _cell_chunks(n_cells, n_genes)
        covered = np.concatenate([np.arange(n_cells)[s] for s in chunks])
        np.testing.assert_array_equal(covered, np.arange(n_cells))
        assert all(s.stop > s.start for s in chunks)


def test_zero_mu_gives_zero(mu_theta):
    _, theta = mu_theta
    mu = np.zeros((10, 60), dtype=np.float32)
    out = mc_log1p_norm(mu, theta, 5, TARGET, scale=np.ones(10), seed=0)
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_invalid_arguments(mu_theta):
    mu, theta = mu_theta
    with pytest.raises(ValueError, match="n_samples"):
        mc_log1p_norm(mu, theta, 0, TARGET)
    with pytest.raises(ValueError, match="theta"):
        mc_log1p_norm(mu, theta[:5], 2, TARGET)
    with pytest.raises(ValueError, match="scale"):
        mc_log1p_norm(mu, theta, 2, TARGET, scale=np.ones(3))


def test_default_threads_respects_cpu_affinity(monkeypatch):
    """Thread count must follow CPU affinity, not the host core count."""
    from cellpin.models.utils import nb_sampling

    monkeypatch.setattr(nb_sampling.os, "cpu_count", lambda: 256)
    monkeypatch.setattr(nb_sampling.os, "sched_getaffinity", lambda _: set(range(2)))
    assert nb_sampling._default_threads() == 2

    monkeypatch.setattr(nb_sampling.os, "sched_getaffinity", lambda _: set(range(200)))
    assert nb_sampling._default_threads() == nb_sampling._MAX_THREADS

    # Platforms without sched_getaffinity fall back to cpu_count.
    monkeypatch.delattr(nb_sampling.os, "sched_getaffinity")
    assert nb_sampling._default_threads() == nb_sampling._MAX_THREADS
