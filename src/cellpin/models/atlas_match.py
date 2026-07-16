"""Atlas-matching encoder.

A deterministic, decoder-free network that maps a **fixed**, augmented gene
panel onto a precomputed atlas embedding (e.g. scVI latent space).  Unlike the
two-view VAE in :mod:`cellpin.models.cellpin_model`, this network carries no
reconstruction, KL, or library machinery — its only job is to reproduce the
reference embedding geometry from a limited panel.

Architecture (compact but modern):

* per-gene input standardisation on ``log1p`` counts (registered buffers),
* a linear stem ``n_panel → n_hidden``,
* ``n_blocks`` pre-norm residual **GEGLU** blocks with LayerScale and
  stochastic depth (DropPath),
* a final norm + linear head to ``emb_dim`` in *standardised* target space.

Target de-standardisation (``ẑ · σ + μ``) is folded into :meth:`predict` and
the de-norm buffers travel with the ``state_dict``.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def per_dim_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-dimension coefficient of determination ``R²``.

    Both tensors are ``(N, D)`` embeddings (any space, but typically the
    *standardised* target space the net predicts in).  Returns a ``(D,)`` tensor
    of ``R² = 1 − SS_res/SS_tot`` per embedding dimension.  ``R² = 1`` is a
    perfect pointwise fit; ``0`` matches a constant (the per-dim mean); negatives
    are worse than that.  This is the metric that literally means "reproduce the
    embedding coordinate", as opposed to neighbour-structure metrics.
    """
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp_min(1e-8)
    return 1.0 - ss_res / ss_tot


def mmd_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Maximum Mean Discrepancy with a multi-scale Gaussian kernel.

    Measures the distance between the distributions of ``x`` and ``y`` without
    requiring any per-sample correspondence.  Uses three bandwidths
    (0.5, 1.0, 2.0) suited to standardised embedding space (unit variance per
    dimension).

    Only gradients w.r.t. ``x`` are meaningful when ``y`` is detached — which
    is the intended use in ``finetune_spatial``: push spatial predictions toward
    the scRNA distribution without pulling the scRNA anchor off its targets.

    Both tensors are ``(B, D)``.  Returns a scalar.
    """
    dxx = torch.cdist(x, x).pow(2)
    dyy = torch.cdist(y, y).pow(2)
    dxy = torch.cdist(x, y).pow(2)
    loss = x.new_zeros(1).squeeze()
    for bw in (0.5, 1.0, 2.0):
        s = 2.0 * bw ** 2
        loss = loss + torch.exp(-dxx / s).mean() + torch.exp(-dyy / s).mean() - 2.0 * torch.exp(-dxy / s).mean()
    return loss / 3.0


def dist_match_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable surrogate for kNN-overlap: match pairwise distances.

    ``kNN-overlap`` itself is non-differentiable (hard ``topk`` + set
    intersection).  Its intent — "cells close in the atlas stay close in the
    prediction" — has a simple differentiable form: ask every within-batch
    pairwise distance in ``pred`` to equal the corresponding atlas distance.
    Preserving distances preserves neighbourhoods.

    Both tensors are ``(B, D)`` (standardised target space).  Returns the MSE
    between the two flattened pairwise-distance vectors.  Complements ``distill``
    (absolute coordinates) by pinning *relative* geometry; mostly helps early,
    before coordinates are nailed.
    """
    return F.mse_loss(torch.pdist(pred), torch.pdist(target))


def soft_centroid_loss(
    pred: torch.Tensor,
    centroids: torch.Tensor,
    type_idx: torch.Tensor,
    confidence: torch.Tensor | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Prototypical (soft nearest-centroid) pull toward per-type atlas centroids.

    Unlike a direct ``MSE(pred, centroid)`` regression — which forces every
    cell onto the *exact* coordinate of its assigned type's mean, regardless of
    how confidently it belongs there — this treats the centroids as class
    prototypes and asks only that ``pred`` be *closer* (in cosine similarity)
    to its own type's centroid than to the others.  Gradients vanish once that
    margin is comfortable, so cells already near the right neighbourhood stop
    being pushed once separated from competing types, and pseudo-labels from
    ``label_transfer`` are treated as soft evidence rather than a hard target.

    Args:
        pred: ``(B, D)`` spatial predictions (standardised atlas space).
        centroids: ``(n_types, D)`` per-type atlas centroids.
        type_idx: ``(B,)`` long tensor of assigned type indices (row into
            ``centroids``); every entry must be a valid row (no ``-1``).
        confidence: Optional ``(B,)`` per-cell weight in ``[0, 1]`` (e.g.
            ``label_transfer`` max-class probability). Cells with lower
            confidence contribute proportionally less to the loss instead of
            being pulled just as hard as confidently-assigned ones.
        temperature: Softmax temperature over cosine-similarity logits; lower
            sharpens the pull toward the assigned centroid, higher softens it.

    Returns:
        Scalar loss (confidence-weighted mean cross-entropy).
    """
    logits = F.cosine_similarity(pred.unsqueeze(1), centroids.unsqueeze(0), dim=-1) / temperature
    per_cell = F.cross_entropy(logits, type_idx, reduction="none")
    if confidence is not None:
        weight = confidence.clamp(0.0, 1.0)
        return (per_cell * weight).sum() / weight.sum().clamp_min(1e-8)
    return per_cell.mean()


@torch.no_grad()
def knn_overlap(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int = 15,
    max_cells: int = 5000,
    seed: int = 0,
) -> float:
    """Mean fraction of each cell's ``k`` atlas neighbours recovered in ``pred``.

    For every cell, the ``k`` nearest neighbours are found in ``target`` space and
    in ``pred`` space (Euclidean, self excluded); the metric is the average
    fraction of the target neighbour set that also appears in the predicted
    neighbour set.  ``1.0`` means neighbourhoods are perfectly preserved.  This
    mirrors what ``sc.pp.neighbors`` + UMAP shows qualitatively, as a single
    number.

    The cost is ``O(n²)``; ``n`` is randomly subsampled to ``max_cells`` for a
    stable, cheap estimate.  Returns ``nan`` if fewer than two cells are present.
    """
    n = target.shape[0]
    if n < 2:
        return float("nan")
    if n > max_cells:
        gen = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=gen)[:max_cells]
        pred, target = pred[idx], target[idx]
        n = max_cells
    k = min(k, n - 1)

    def neighbours(x: torch.Tensor) -> torch.Tensor:
        d = torch.cdist(x, x)
        d.fill_diagonal_(float("inf"))
        return d.topk(k, largest=False).indices  # (n, k)

    nbr_target = neighbours(target)
    nbr_pred = neighbours(pred)

    rows = torch.arange(n).unsqueeze(1).expand(-1, k)
    membership = torch.zeros(n, n, dtype=torch.bool)
    membership[rows, nbr_target] = True  # True where j is a true neighbour of i
    hit = membership[rows, nbr_pred]  # (n, k) — predicted neighbour is a true one
    return hit.float().mean().item()


# ---------------------------------------------------------------------------
# Exponential moving average of weights
# ---------------------------------------------------------------------------


class LinearRampCallback(pl.Callback):
    """Linearly ramp a named ``pl_module`` attribute from 0 → 1 after a warmup phase.

    Sets ``getattr(pl_module, attr)`` at the start of each training epoch. During
    the first ``warmup_frac`` of epochs the value is 0; it then increases
    linearly to 1 over the remaining epochs. Generic building block for any loss
    weight that should ease in rather than apply at full strength from epoch 0 —
    e.g. augmentation strength (``_aug_strength``, read by
    ``_poisson_resample_panel``/``_mixup_panel``) or the spatial-alignment loss
    weight in ``finetune_spatial`` (``_centroid_weight_scale``), which otherwise
    fights the panel encoder's still-forming predictions from the first batch.

    Args:
        attr: Name of the float attribute to set on ``pl_module`` each epoch.
        warmup_frac: Fraction of total epochs to hold the value at zero.
    """

    def __init__(self, attr: str, warmup_frac: float = 0.25) -> None:
        super().__init__()
        self.attr = attr
        self.warmup_frac = warmup_frac

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        progress = trainer.current_epoch / max(1, trainer.max_epochs)
        strength = max(0.0, (progress - self.warmup_frac) / (1.0 - self.warmup_frac))
        setattr(pl_module, self.attr, float(min(1.0, strength)))


class EMACallback(pl.Callback):
    """Maintain an exponential moving average (EMA) of the trainable weights.

    Regression objectives on a plateau respond well to weight averaging: the EMA
    weights are smoother than any single SGD iterate and usually give a small,
    reliable bump with no downside.

    The averaged weights are swapped in for **validation** (so logged metrics and
    any checkpoint saved during validation reflect the EMA) and swapped back out
    at the start of the next training epoch (so optimisation continues on the raw
    weights).  At the end of ``fit`` the EMA weights are left in the model
    permanently.  Only parameters with ``requires_grad`` are tracked, so a frozen
    backbone (e.g. the VAE in ``emb_match``) is ignored automatically.

    Args:
        decay: EMA decay; higher = slower/smoother. ``0`` disables tracking.
    """

    def __init__(self, decay: float = 0.999) -> None:
        super().__init__()
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.shadow = {
            name: p.detach().clone()
            for name, p in pl_module.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs) -> None:  # noqa: ANN001
        for name, p in pl_module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.shadow:  # sanity-check validation runs before on_train_start
            return
        self.backup = {}
        for name, p in pl_module.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.backup:  # restore raw weights for the upcoming training epoch
            for name, p in pl_module.named_parameters():
                if name in self.backup:
                    p.data.copy_(self.backup[name])
            self.backup = {}

    @torch.no_grad()
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.shadow:
            return
        for name, p in pl_module.named_parameters():  # leave EMA weights in place
            if name in self.shadow:
                p.data.copy_(self.shadow[name])
        self.backup = {}


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth (drops whole residual branches)."""
    if drop_prob <= 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


class GEGLUBlock(nn.Module):
    """Pre-norm residual block with a gated (GEGLU) feed-forward.

    GEGLU: ``out = (W_v x) ⊙ gelu(W_g x)`` — the gating that powers modern
    transformer FFNs, giving more expressivity per parameter than a plain MLP.
    LayerScale + DropPath stabilise deep residual stacks.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        hidden = int(dim * expansion)
        self.norm = nn.LayerNorm(dim)
        self.proj_in = nn.Linear(dim, hidden * 2)  # value + gate
        self.proj_out = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.drop_path_rate = drop_path_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        value, gate = self.proj_in(h).chunk(2, dim=-1)
        h = self.proj_out(self.dropout(value * F.gelu(gate)))
        return x + drop_path(self.gamma * h, self.drop_path_rate, self.training)


class AtlasMatchNet(nn.Module):
    """Map a fixed augmented panel to a frozen atlas embedding.

    Args:
        n_panel: Number of panel genes (input dimension).
        emb_dim: Atlas embedding dimension (output).
        n_hidden: Width of the residual trunk.
        n_blocks: Number of GEGLU residual blocks.
        expansion: FFN expansion factor inside each block.
        dropout: Dropout inside blocks.
        drop_path_rate: Maximum stochastic-depth rate (linearly scaled by depth).
        layer_scale_init: Initial LayerScale value.
        log_input: Apply ``log1p`` to the panel before standardisation.
    """

    def __init__(
        self,
        n_panel: int,
        emb_dim: int,
        n_hidden: int = 256,
        n_blocks: int = 4,
        expansion: float = 2.0,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-3,
        log_input: bool = True,
    ) -> None:
        super().__init__()
        self.n_panel = n_panel
        self.emb_dim = emb_dim
        self.log_input = log_input

        # Input standardisation (filled by set_input_stats); identity by default.
        self.register_buffer("x_mean", torch.zeros(n_panel))
        self.register_buffer("x_std", torch.ones(n_panel))
        # Target de-standardisation (filled by set_target_stats); identity default.
        self.register_buffer("z_mu", torch.zeros(emb_dim))
        self.register_buffer("z_sigma", torch.ones(emb_dim))

        self.stem = nn.Linear(n_panel, n_hidden)
        # Linearly increasing drop-path per depth (stochastic-depth schedule).
        dprs = [drop_path_rate * i / max(1, n_blocks - 1) for i in range(n_blocks)]
        self.blocks = nn.ModuleList(
            GEGLUBlock(
                n_hidden,
                expansion=expansion,
                dropout=dropout,
                drop_path_rate=dpr,
                layer_scale_init=layer_scale_init,
            )
            for dpr in dprs
        )
        self.norm_out = nn.LayerNorm(n_hidden)
        self.head = nn.Linear(n_hidden, emb_dim)

    # -- statistics --------------------------------------------------------
    @torch.no_grad()
    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set per-gene input mean/std (computed on ``log1p`` counts)."""
        self.x_mean.copy_(mean.to(self.x_mean))
        self.x_std.copy_(std.clamp_min(1e-6).to(self.x_std))

    @torch.no_grad()
    def set_target_stats(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        """Set per-dimension atlas-embedding mean/std for de-standardisation."""
        self.z_mu.copy_(mu.to(self.z_mu))
        self.z_sigma.copy_(sigma.clamp_min(1e-6).to(self.z_sigma))

    def standardize_target(self, z: torch.Tensor) -> torch.Tensor:
        """Map an atlas embedding into the standardised space the net predicts."""
        return (z - self.z_mu) / self.z_sigma

    # -- forward -----------------------------------------------------------
    def forward(self, x_panel: torch.Tensor) -> torch.Tensor:
        """Return the predicted embedding in *standardised* space."""
        x = torch.log1p(x_panel) if self.log_input else x_panel
        x = (x - self.x_mean) / self.x_std
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm_out(h))

    @torch.no_grad()
    def predict(self, x_panel: torch.Tensor) -> torch.Tensor:
        """Return the predicted embedding in the original atlas space."""
        return self(x_panel) * self.z_sigma + self.z_mu
