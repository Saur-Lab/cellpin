"""Monte-Carlo estimation of ``E[log1p(norm(X))]`` for ``X ~ NB(mu, theta)``.

Used by :meth:`cellpin.CellPin.impute` to build ``layers['imputed_norm']``.
Because ``log1p`` is concave, Jensen's inequality gives
``log1p(norm(E[X])) > E[log1p(norm(X))]``; drawing counts *inside* the
transform removes that bias.

The estimator is parallel over cells, so the work is split into
cell chunks that are sampled independently.  Two backends are available:

* ``numpy`` — chunks run on a thread pool.  :class:`numpy.random.Generator`
  releases the GIL while sampling, so this scales close to linearly with cores.
* ``torch`` — chunks run sequentially on a CUDA device, using the
  Gamma-Poisson representation of the NegativeBinomial.

Both backends draw one independent RNG stream per chunk, and the chunking is a
function of the input shape alone, so for a fixed seed the result is
reproducible and independent of thread count.  The two backends use different
samplers and therefore agree only up to Monte-Carlo error.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

# Elements (cells x genes) per chunk.  Each chunk holds a handful of arrays of
# this size, so ~2M elements keeps a worker's working set around 50 MB.
_CHUNK_ELEMENTS = 2_000_000

# Lower bound on the number of chunks, so a thread pool this wide stays busy.
_MIN_CHUNKS = 64

# Floor on mu before forming the NB success probability / Gamma rate.
_MU_FLOOR = 1e-8

# Above this the thread pool stops paying for itself.
_MAX_THREADS = 32


def _default_threads() -> int:
    """Return the number of CPUs this process may actually run on.

    ``os.cpu_count()`` reports the host's cores and ignores CPU affinity, so
    under a scheduler that pins jobs to a cpuset (Slurm, containers) it
    over-reports badly: a 2-core allocation on a 36-core node still reads 36,
    and the pool then oversubscribes its cores by an order of magnitude.
    ``sched_getaffinity`` reports the real allocation, but is Linux-only.
    """
    try:
        return min(_MAX_THREADS, len(os.sched_getaffinity(0)))
    except AttributeError:  # not available on macOS / Windows
        return min(_MAX_THREADS, os.cpu_count() or 1)


def _cell_chunks(n_cells: int, n_genes: int) -> list[slice]:
    """Split the cell axis into chunks bounded by :data:`_CHUNK_ELEMENTS`.

    Chunking on the *cell* axis (never the gene axis) is required: the
    library-size branch normalises by a per-cell sum over all genes.

    Depends on the input shape only, never on the worker count, so that a
    seeded run reproduces regardless of how many threads it gets.
    """
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // max(n_genes, 1))
    rows_per_chunk = min(rows_per_chunk, max(1, -(-n_cells // _MIN_CHUNKS)))
    n_chunks = max(1, -(-n_cells // rows_per_chunk))
    bounds = np.linspace(0, n_cells, n_chunks + 1).astype(int)
    return [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True) if hi > lo]


def _chunk_numpy(
    mu: np.ndarray,
    theta: np.ndarray,
    scale: np.ndarray | None,
    n_samples: int,
    norm_target_sum: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Accumulate ``mean_k log1p(norm(draw_k))`` for one chunk of cells."""
    # p = theta / (theta + mu); float64 because it parameterises the sampler.
    p = theta / (theta + np.clip(mu, _MU_FLOOR, None).astype(np.float64))
    acc = np.zeros(mu.shape, dtype=np.float32)
    for _ in range(n_samples):
        draw = rng.negative_binomial(theta, p).astype(np.float32)
        if scale is not None:
            normed = draw * scale[:, np.newaxis]
        else:
            lib = draw.sum(axis=1, keepdims=True).clip(1e-12)
            normed = draw * (norm_target_sum / lib)
        acc += np.log1p(normed)
    return acc / n_samples


def _chunk_torch(
    mu: np.ndarray,
    theta: torch.Tensor,
    scale: np.ndarray | None,
    n_samples: int,
    norm_target_sum: float,
    generator: torch.Generator,
    device: torch.device,
) -> np.ndarray:
    """GPU counterpart of :func:`_chunk_numpy` via the Gamma-Poisson mixture.

    ``X ~ NB(mu, theta)`` iff ``X | lam ~ Poisson(lam)`` with
    ``lam ~ Gamma(shape=theta, rate=theta/mu)``.
    """
    mu_t = torch.from_numpy(mu).to(device=device, dtype=torch.float32)
    concentration = theta.expand_as(mu_t).contiguous()
    rate = theta / mu_t.clamp_min(_MU_FLOOR)
    scale_t = None if scale is None else torch.from_numpy(scale).to(device=device, dtype=torch.float32)[:, None]

    acc = torch.zeros_like(mu_t)
    for _ in range(n_samples):
        lam = torch._standard_gamma(concentration, generator) / rate
        draw = torch.poisson(lam, generator=generator)
        if scale_t is not None:
            normed = draw * scale_t
        else:
            lib = draw.sum(dim=1, keepdim=True).clamp_min(1e-12)
            normed = draw * (norm_target_sum / lib)
        acc += torch.log1p(normed)
    return (acc / n_samples).cpu().numpy()


def mc_log1p_norm(
    mu: np.ndarray,
    theta: np.ndarray,
    n_samples: int,
    norm_target_sum: float,
    scale: np.ndarray | None = None,
    seed: int | None = None,
    device: torch.device | str | None = None,
    n_threads: int | None = None,
) -> np.ndarray:
    """Estimate ``E[log1p(norm(X))]`` with ``X ~ NB(mu, theta)`` per element.

    Args:
        mu: Expected counts ``(n_cells, n_genes)``.
        theta: Per-gene inverse dispersion ``(n_genes,)``, strictly positive.
        n_samples: Number of NB draws averaged per element.
        norm_target_sum: Target total counts after normalisation.
        scale: Per-cell normalisation factor ``(n_cells,)`` for area-based
            normalisation.  When ``None``, each draw is normalised by its own
            library size.
        seed: Seed for the per-chunk RNG streams.  ``None`` draws entropy from
            the OS, matching unseeded behaviour.
        device: Torch device for the GPU backend.  ``None`` or a CPU device
            selects the threaded numpy backend, which is faster on CPU because
            torch's CPU RNG is single-threaded.
        n_threads: Worker threads for the numpy backend.  Defaults to the
            number of CPUs this process is actually allowed to use (CPU
            affinity, so Slurm and container allocations are respected),
            capped at 32.  Ignored by the torch backend.

    Returns:
    -------
        Float32 array ``(n_cells, n_genes)``.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")

    n_cells, n_genes = mu.shape
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (n_genes,):
        raise ValueError(f"theta must have shape ({n_genes},), got {theta.shape}")
    if scale is not None:
        scale = np.asarray(scale, dtype=np.float32)
        if scale.shape != (n_cells,):
            raise ValueError(f"scale must have shape ({n_cells},), got {scale.shape}")

    mu = np.ascontiguousarray(mu, dtype=np.float32)

    device = torch.device(device) if device is not None else None
    use_torch = device is not None and device.type != "cpu"

    n_workers = 1 if use_torch else (n_threads or _default_threads())
    chunks = _cell_chunks(n_cells, n_genes)
    seeds = np.random.SeedSequence(seed).spawn(len(chunks))

    out = np.empty((n_cells, n_genes), dtype=np.float32)

    if use_torch:
        theta_t = torch.from_numpy(theta).to(device=device, dtype=torch.float32)
        for sl, seed_seq in zip(chunks, seeds, strict=True):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed_seq.generate_state(2, dtype=np.uint64)[0] >> 1))
            out[sl] = _chunk_torch(
                mu[sl],
                theta_t,
                None if scale is None else scale[sl],
                n_samples,
                norm_target_sum,
                generator,
                device,
            )
        return out

    def _run(job: tuple[slice, np.random.SeedSequence]) -> None:
        sl, seed_seq = job
        out[sl] = _chunk_numpy(
            mu[sl],
            theta,
            None if scale is None else scale[sl],
            n_samples,
            norm_target_sum,
            np.random.default_rng(seed_seq),
        )

    jobs = list(zip(chunks, seeds, strict=True))
    if n_workers == 1 or len(jobs) == 1:
        for job in jobs:
            _run(job)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(_run, jobs))

    return out
