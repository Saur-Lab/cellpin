"""CellPin Model.

Training pipeline
-----------------
**Stage 1 — pretrain** (full-gene path only):

* ELBO = NB/ZINB reconstruction + KL(q(z)||p(z)) + KL(q(l)||p(l))

**Stage 2 — main training** (both views):

* ELBO via the **panel encoder** (imputation objective).
* Invariance loss: KL-distillation/MSE + soft nearest-neighbour (panel → full).

Additional features
-------------------
* KL annealing (linear warm-up over ``kl_warmup_epochs`` epochs).
* Per-stage configurable loss weights.
* ``get_cell_embedding``, ``embed_and_impute`` API.
* ``fit()`` convenience wrapper: pretrain → train in one call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from cellpin._sdata_utils import _resolve_sdata
from cellpin.dataset import scAnnDataset
from cellpin.models.utils import (
    build_data_loaders,
    load_config_and_checkpoint,
    mc_log1p_norm,
    save_checkpoint,
)
from cellpin.models.vae import CellPinVAE
from cellpin.pl import PlotAccessor
from cellpin.tl import TLAccessor
from cellpin.training import CellPinTrainer


def soft_nn_loss(
    z_panel: torch.Tensor,
    z_full: torch.Tensor,
    temperature: float | torch.Tensor = 0.1,
) -> torch.Tensor:
    """Symmetric soft nearest-neighbour alignment between latent spaces.

    Uses cross-entropy over cosine-similarity logits with same-index positives.
    Accepts a scalar float or a learnable tensor for ``temperature``.
    """
    z_panel = F.normalize(z_panel, p=2, dim=-1)
    z_full = F.normalize(z_full, p=2, dim=-1)

    logits = (z_panel @ z_full.T) / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_pf = F.cross_entropy(logits, targets)
    loss_fp = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_pf + loss_fp)


class CellPin(pl.LightningModule):
    """CellPin: hybrid two-view VAE for single-cell and spatial transcriptomics.

    * Model construction (:class:`~cellpin.models.vae.CellPinVAE`).
    * Two-stage training (pretrain → main).
    * ELBO computation with KL annealing.
    * Invariance regularisation (KL-distillation/MSE + SNN).
    * Inference / imputation API.

    Args:
        sc_dataset: Training :class:`~cellpin.dataset.scAnnDataset` used to
            infer gene counts and panel size.
        config: Hyper-parameter dict, path to a YAML file, or ``None`` for
            defaults. Key parameters:

            - ``n_latent`` (192): latent space dimensionality
            - ``n_hidden`` (1024): encoder/decoder hidden width
            - ``encoder_layers`` (16): number of residual blocks per encoder
            - ``reconstruction_loss`` ("nb"): ``"nb"``, ``"zinb"``, ``"poisson"``, ``"normal"``, ``"zin"``
            - ``distillation_mode`` ("mse"): ``"kl"`` or ``"mse"``
            - ``reconstruct_panel`` (True): if ``False``, reconstruction loss on non-panel genes only
            - ``kl_warmup_epochs`` (20): epochs for linear KL annealing from 0 to ``kl_weight``
            - ``lambda_inv`` (20.0): weight on the invariance (distillation + SNN) loss
            - ``exclude_panel`` (False): if ``True``, full encoder sees panel genes zeroed out
            - ``lr`` (0.00021): AdamW learning rate

            Full defaults in ``configs/cellpin_config.yaml``.
        checkpoint: Path to a ``.pt`` checkpoint to load weights from.
    """

    def __init__(
        self,
        sc_dataset: scAnnDataset,
        config: dict[str, Any] | str | Path | None = None,
        checkpoint: str | Path | None = None,
    ):
        super().__init__()

        params, loaded_state_dict = load_config_and_checkpoint(config, checkpoint)
        for k, v in params.items():
            setattr(self, k, v)

        # Gene info — derived from the dataset and embedded in hparams so that
        # a saved checkpoint is self-contained (useful for standalone scripts).
        self.num_genes = int(sc_dataset.X.shape[1])
        self.gene_names = [str(g) for g in sc_dataset.gene_names.tolist()]
        self.n_panel_genes = int(sc_dataset._panel_mask.sum().item())
        # Panel gene names in the order the panel encoder expects them:
        # boolean mask applied to full_expr → ascending sc_adata position order.
        self.panel_gene_names: list[str] = [
            self.gene_names[i] for i, m in enumerate(sc_dataset._panel_mask.tolist()) if m
        ]
        params["_gene_names"] = self.gene_names
        params["_panel_gene_names"] = self.panel_gene_names
        params["_n_input_full"] = self.num_genes
        params["_n_input_panel"] = self.n_panel_genes

        self.save_hyperparameters(params)

        # Loss weights
        # Pretrain stage
        self.loss_weights_pretrain: dict[str, float] = {
            "kl_weight": float(getattr(self, "kl_weight_pretrain", 1.0)),
            "recon": float(getattr(self, "lambda_recon_pretrain", 1.0)),
            "inv": 0.0,
        }
        # Main training stage
        self.loss_weights_train: dict[str, float] = {
            "kl_weight": float(getattr(self, "kl_weight", 1.0)),
            "recon": float(getattr(self, "lambda_recon", 1.0)),
            "inv": float(getattr(self, "lambda_inv", 1.0)),
            "snn": float(getattr(self, "lambda_snn", 0.1)),
            "distill": float(getattr(self, "lambda_distill", 1.0)),
        }

        self.kl_warmup_epochs = int(getattr(self, "kl_warmup_epochs", 0))

        # n_batch: config takes precedence; fall back to dataset's encoded batch count
        _n_batch = int(getattr(self, "n_batch", 0))
        if _n_batch == 0 and hasattr(sc_dataset, "n_batch"):
            _n_batch = int(sc_dataset.n_batch)

        self.vae = CellPinVAE(
            n_input_full=self.num_genes,
            n_input_panel=self.n_panel_genes,
            panel_idx=getattr(sc_dataset, "panel_idx", None),
            use_panel_only=bool(getattr(self, "use_panel_only", True)),
            n_batch=_n_batch,
            n_hidden=int(getattr(self, "n_hidden", 128)),
            n_latent=int(getattr(self, "n_latent", 32)),
            n_layers_encoder=int(getattr(self, "encoder_layers", 4)),
            n_layers_decoder=int(getattr(self, "decoder_layers", 2)),
            dropout_rate=float(getattr(self, "encoder_dropout", 0.0)),
            drop_path_rate=float(getattr(self, "drop_path_rate", 0.0)),
            ffn_expansion=float(getattr(self, "ffn_expansion", 2.0)),
            layer_scale_init=float(getattr(self, "layer_scale_init", 1e-3)),
            dispersion=str(getattr(self, "dispersion", "gene")),
            log_variational=bool(getattr(self, "log_variational", True)),
            reconstruction_loss=str(getattr(self, "reconstruction_loss", "zinb")),
            latent_distribution=str(getattr(self, "latent_distribution", "normal")),
            # The noise is injected inside Encoder.forward() on both z_encoder_full
            # and z_encoder_panel, training-mode only.
            input_noise_std=float(getattr(self, "encoder_noise_std", 0.0)),
            exclude_panel=bool(getattr(self, "exclude_panel", False)),
            batch_infer_mode=str(getattr(self, "batch_infer_mode", "mean_onehot")),
        )

        if loaded_state_dict is not None:
            self.load_state_dict(loaded_state_dict, strict=False)

        # Learnable SNN temperature (log-parameterised so it stays positive).
        # At inference the effective temperature = exp(param).clamp(0.01, 1.0).
        _t_init = float(getattr(self, "snn_temperature_init", 0.07))
        self.snn_temperature = nn.Parameter(torch.tensor(_t_init).log())

        # When False: reconstruction loss is computed on non-panel genes only.
        # At inference, observed panel values are copied into the decoder output.
        self._reconstruct_panel: bool = bool(getattr(self, "reconstruct_panel", True))

        self._training_stage: Literal["pretrain", "main"] = "main"
        self._pretrain_completed: bool = False
        self._freeze_pretrained_in_main: bool = True
        self._decoder_warm_unfreeze_epoch: int = int(getattr(self, "decoder_warm_unfreeze_epoch", -1))
        self._decoder_unfrozen: bool = True

        # Log directories set after training — used by model.pl.losses()
        self._pretrain_output_dir: Path | None = None
        self._train_output_dir: Path | None = None

        self.pl = PlotAccessor(self)
        self.tl = TLAccessor(self)

    def save(self, path: str | Path) -> None:
        """Serialise model weights and hyper-parameters to a ``.pt`` file.

        Args:
            path: Destination file path.
        """
        save_checkpoint(Path(path), self.state_dict(), dict(self.hparams))

    def _get_loss_weights(self, stage: str) -> dict[str, float]:
        """Return loss weights for the given training stage.

        Args:
            stage: ``'pretrain'`` or ``'main'``.

        Returns:
        -------
            Dict with keys ``'kl_weight'``, ``'recon'``, ``'inv'``.
        """
        return self.loss_weights_pretrain if stage == "pretrain" else self.loss_weights_train

    def set_stage_loss_weights(self, stage: str, **weights: float) -> None:
        """Programmatically update loss weights for a stage.

        Useful for sweeps or ablations.

        Args:
            stage: ``'pretrain'`` or ``'main'``.
            **weights: Key-value overrides, e.g. ``inv=2.0``, ``recon=1.5``.

        Raises:
        ------
            KeyError: For unknown weight keys.

        Example::

            model.set_stage_loss_weights("main", inv=2.0, recon=1.5)
        """
        target = self.loss_weights_pretrain if stage == "pretrain" else self.loss_weights_train
        for k, v in weights.items():
            if k not in target:
                raise KeyError(f"Unknown weight '{k}'. Valid keys: {list(target.keys())}")
            target[k] = float(v)

    def _set_stage_trainability(
        self,
        stage: Literal["pretrain", "main"],
        freeze_pretrained: bool = True,
        decoder_trainable_override: bool | None = None,
    ) -> None:
        """Configure trainable modules per stage.

        Stage 1 (pretrain): train full encoder + decoder + library encoder,
        keep panel encoder frozen.

        Stage 2 (main): train panel encoder (+ library encoder), and freeze
        pretrained full encoder + decoder when ``freeze_pretrained=True``.
        """
        pretrain_mode = stage == "pretrain"

        full_trainable = pretrain_mode or not freeze_pretrained
        decoder_trainable = pretrain_mode or not freeze_pretrained
        if decoder_trainable_override is not None:
            decoder_trainable = decoder_trainable_override
        panel_trainable = not pretrain_mode
        library_trainable = True

        for p in self.vae.z_encoder_full.parameters():
            p.requires_grad = full_trainable
        for p in self.vae.z_encoder_panel.parameters():
            p.requires_grad = panel_trainable
        for p in self.vae.l_encoder.parameters():
            p.requires_grad = library_trainable
        for p in self.vae.decoder.parameters():
            p.requires_grad = decoder_trainable

    def on_train_epoch_start(self) -> None:
        """Warm-unfreeze decoder parameters when scheduled."""
        if self._training_stage not in ("main",):
            return
        if self._decoder_warm_unfreeze_epoch < 0 or self._decoder_unfrozen:
            return
        if self.current_epoch < self._decoder_warm_unfreeze_epoch:
            return

        for p in self.vae.decoder.parameters():
            p.requires_grad = True
        self._decoder_unfrozen = True

        # Decoder params are already part of the optimizer when warm-unfreeze
        # is scheduled (see configure_optimizers); toggling requires_grad is enough.
        self.print(f"Decoder warm-unfrozen at epoch {self.current_epoch}.")

    def _kl_annealing_weight(self) -> float:
        """Compute the current KL annealing multiplier (linear warm-up).

        Returns:
        -------
            Float in ``[0.0, 1.0]``.
        """
        if self.kl_warmup_epochs <= 0:
            return 1.0
        return min(1.0, self.current_epoch / self.kl_warmup_epochs)

    # ------------------------------------------------------------------
    # Panel augmentation
    # ------------------------------------------------------------------

    def _mixup_panel(self, x_panel: torch.Tensor) -> torch.Tensor:
        """Intra-batch contamination mixup for the panel encoder input; mimics missegmentation in spatial data.

        Each cell is blended with a randomly permuted cell from the same
        minibatch.  The contamination fraction ``alpha`` is sampled
        uniformly in ``[0, panel_mixup_alpha]`` per cell so the encoder
        learns to be robust to partial ambient / spot contamination.

        Applied **only** during training; returns ``x_panel`` unchanged at
        eval time or when ``panel_mixup_alpha == 0``.

        Config key: ``panel_mixup_alpha`` (float, default ``0.0`` = disabled).
        """
        max_alpha = float(getattr(self, "panel_mixup_alpha", 0.0))
        if not self.training or max_alpha <= 0.0:
            return x_panel

        B = x_panel.size(0)
        alpha = torch.rand(B, 1, device=x_panel.device) * max_alpha
        perm = torch.randperm(B, device=x_panel.device)
        return (1.0 - alpha) * x_panel + alpha * x_panel[perm]

    def _poisson_resample_panel(self, x_panel: torch.Tensor) -> torch.Tensor:
        """Simulate spatial capture efficiency via Poisson resampling.

        Draws a random capture efficiency ``eff ~ Uniform(1 - rate, 1)`` for
        the whole batch, scales the panel counts, then resamples from a Poisson
        distribution. Mimics the lower mRNA capture typical of spatial
        platforms compared to scRNA-seq.

        Applied **only** during training; returns ``x_panel`` unchanged at
        eval time or when ``poisson_resample_rate == 0``.

        Config key: ``poisson_resample_rate`` (float in ``[0, 1]``,
        default ``0.0`` = disabled).  A value of ``0.4`` draws capture
        efficiency uniformly from ``[0.6, 1.0]``.
        """
        rate = float(getattr(self, "poisson_resample_rate", 0.0))
        if not self.training or rate <= 0.0:
            return x_panel
        eff = 1.0 - torch.rand(1, device=x_panel.device) * rate
        return torch.poisson(x_panel * eff)

    @staticmethod
    def _pearson_loss(px_rate: torch.Tensor, x_full: torch.Tensor) -> torch.Tensor:
        """Per-gene Pearson correlation loss: ``1 - mean_g r(px_rate[:, g], x[:, g])``.

        Computes Pearson r across cells for every gene in the batch, then
        returns ``1 - mean(r)``.  Perfect per-gene correlation → loss = 0;
        anti-correlation → loss = 2. Mainly for logging, keep loss weight low.

        Args:
            px_rate: Predicted NB rate ``(batch, n_genes)``.
            x_full:  Observed counts ``(batch, n_genes)``.

        Returns:
        -------
            Scalar loss tensor.
        """
        p = px_rate - px_rate.mean(dim=0, keepdim=True)
        t = x_full - x_full.mean(dim=0, keepdim=True)
        num = (p * t).sum(dim=0)  # (G,)
        denom = p.norm(dim=0) * t.norm(dim=0) + 1e-8  # (G,)
        r = num / denom  # (G,) in [-1, 1]
        return 1.0 - r.mean()

    def _mask_recon_to_no_panel(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice reconstruction tensors to non-panel genes only.

        Used when ``reconstruct_panel=False`` so the NB/ZINB loss is computed
        exclusively on genes that are *not* observed as panel input.  Panel gene
        positions are absent from the returned tensors.

        ``px_r`` shape depends on ``dispersion``:

        * ``'gene'``      — ``(n_genes,)``      → ``(n_no_panel,)``
        * ``'gene-cell'`` — ``(batch, n_genes)`` → ``(batch, n_no_panel)``
        """
        mask = ~self.vae.panel_mask  # (n_genes,) bool, True = non-panel
        x_m = x[:, mask]
        rate_m = px_rate[:, mask]
        if self.vae.dispersion == "gene-cell":
            r_m = px_r[:, mask]
        else:  # "gene"
            r_m = px_r[mask]
        drop_m = px_dropout[:, mask] if px_dropout is not None else px_dropout
        return x_m, rate_m, r_m, drop_m

    # Loss computation
    def compute_pretrain_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Pretraining losses (full-gene view only).

        Objective: ELBO on full-gene path.

        Args:
            batch: Must contain ``'full_expr'`` and ``'panel_expr'``.
                Optionally ``'local_l_mean'``, ``'local_l_var'``, ``'batch_index'``.

        Returns:
        -------
            Dict with scalar tensors: ``'loss'``, ``'reconst_loss'``,
            ``'kl_loss'``, ``'kl_l_loss'``.
        """
        x_full = batch["full_expr"]
        x_panel = batch["panel_expr"]
        local_l_mean = batch.get("local_l_mean", torch.zeros(x_full.size(0), 1, device=x_full.device))
        local_l_var = batch.get("local_l_var", torch.ones(x_full.size(0), 1, device=x_full.device))
        batch_index = batch.get("batch_index", None)

        # Augment x_panel before it reaches l_encoder (same augmentations as
        # stage 2 — Poisson resampling for capture-efficiency robustness, then
        # mixup for contamination robustness).
        x_panel = self._poisson_resample_panel(x_panel)
        x_panel = self._mixup_panel(x_panel)

        # Single inference pass — full encoder, panel library
        out = self.vae.inference(
            x_full,
            x_panel=x_panel,
            encoder_view="full",
            batch_index=batch_index,
        )

        # ELBO components — KL against prior
        qz_v_clamped = out["qz_v"].clamp(min=1e-4, max=1e4)
        kl_z = kl(
            Normal(out["qz_m"], qz_v_clamped.sqrt()),
            Normal(torch.zeros_like(out["qz_m"]), torch.ones_like(qz_v_clamped)),
        ).sum(dim=1)

        kl_l = kl(
            Normal(out["ql_m"], out["ql_v"].sqrt()),
            Normal(local_l_mean, local_l_var.sqrt()),
        ).sum(dim=1)

        if self._reconstruct_panel or self.vae.panel_mask is None:
            x_r, rate_r, r_r, drop_r = x_full, out["px_rate"], out["px_r"], out["px_dropout"]
        else:
            x_r, rate_r, r_r, drop_r = self._mask_recon_to_no_panel(
                x_full, out["px_rate"], out["px_r"], out["px_dropout"]
            )
        reconst_loss = self.vae.get_reconstruction_loss(x_r, rate_r, r_r, drop_r).mean()
        kl_loss = kl_z.mean()

        # Optional Pearson correlation loss on the full encoder output
        lambda_pearson = float(getattr(self, "lambda_pearson", 0.0))
        if lambda_pearson > 0.0:
            pearson_loss = self._pearson_loss(out["px_rate"], x_full)
        else:
            pearson_loss = torch.tensor(0.0, device=x_full.device)

        w = self._get_loss_weights("pretrain")
        kl_w = self._kl_annealing_weight() * w["kl_weight"]
        total = (
            w["recon"] * reconst_loss
            + kl_w * kl_loss
            + kl_l.mean()  # library KL always weight-1
            + lambda_pearson * pearson_loss
        )

        return {
            "loss": total,
            "reconst_loss": reconst_loss,
            "kl_loss": kl_loss,
            "kl_l_loss": kl_l.mean(),
            "pearson_loss": pearson_loss,
        }

    def compute_losses(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Main training losses.

        Objective:

        * ELBO via **panel encoder** (imputation-facing objective).
        * Invariance loss: KL-distillation or MSE between full-latent and panel-latent.
        * Soft nearest-neighbour (SNN) alignment.

        A single inference pass per view is made; the ELBO, invariance loss,
        and SNN all share the same sampled ``z`` and means.

        Args:
            batch: Must contain ``'full_expr'`` and ``'panel_expr'``.
                Optionally ``'local_l_mean'``, ``'local_l_var'``, ``'batch_index'``.

        Returns:
        -------
            Dict with scalar tensors: ``'loss'``, ``'reconst_loss'``,
            ``'kl_loss'``, ``'kl_l_loss'``, ``'distill_loss'``,
            ``'snn_loss'``, ``'inv_loss'``, ``'snn_temperature'``,
            ``'pearson_loss'``.
        """
        x_full = batch["full_expr"]
        x_panel = batch["panel_expr"]
        local_l_mean = batch.get("local_l_mean", torch.zeros(x_full.size(0), 1, device=x_full.device))
        local_l_var = batch.get("local_l_var", torch.ones(x_full.size(0), 1, device=x_full.device))
        batch_index = batch.get("batch_index", None)

        # Pass 1: full encoder — no grad, used only as reference
        with torch.no_grad():
            out_full = self.vae.inference(
                x_full,
                x_panel=x_panel,
                encoder_view="full",
                batch_index=batch_index,
            )

        # Pass 2: panel encoder — single pass for ELBO + invariance
        # Poisson resampling first (domain-gap aug), then mixup (contamination aug)
        x_panel = self._poisson_resample_panel(x_panel)
        x_panel = self._mixup_panel(x_panel)  # augment panel input only; x_full unchanged
        out_panel = self.vae.inference(
            x_full,
            x_panel=x_panel,
            encoder_view="panel",
            batch_index=batch_index,
        )

        # ---- ELBO from panel outputs — KL against prior ----
        panel_qz_v_elbo = out_panel["qz_v"].clamp(min=1e-4, max=1e4)
        kl_z = kl(
            Normal(out_panel["qz_m"], panel_qz_v_elbo.sqrt()),
            Normal(torch.zeros_like(out_panel["qz_m"]), torch.ones_like(panel_qz_v_elbo)),
        ).sum(dim=1)

        kl_l = kl(
            Normal(out_panel["ql_m"], out_panel["ql_v"].sqrt()),
            Normal(local_l_mean, local_l_var.sqrt()),
        ).sum(dim=1)

        if self._reconstruct_panel or self.vae.panel_mask is None:
            x_r, rate_r, r_r, drop_r = (x_full, out_panel["px_rate"], out_panel["px_r"], out_panel["px_dropout"])
        else:
            x_r, rate_r, r_r, drop_r = self._mask_recon_to_no_panel(
                x_full, out_panel["px_rate"], out_panel["px_r"], out_panel["px_dropout"]
            )
        reconst_loss = self.vae.get_reconstruction_loss(x_r, rate_r, r_r, drop_r).mean()
        kl_loss = kl_z.mean()

        # ---- Optional Pearson correlation loss ----
        # lambda_pearson=0.0 → disabled (default).
        # loss_pearson = 1 - mean_g pearson(px_rate[:, g], x_full[:, g])
        lambda_pearson = float(getattr(self, "lambda_pearson", 0.0))
        if lambda_pearson > 0.0:
            pearson_loss = self._pearson_loss(out_panel["px_rate"], x_full)
        else:
            pearson_loss = torch.tensor(0.0, device=x_full.device)

        w = self._get_loss_weights("main")

        # ---- Invariance: distill full posterior into panel posterior ----
        # distillation_mode="kl"  → KL(q_panel || q_full)  [default]
        # distillation_mode="mse" → MSE between posterior means (simpler, no variance term)
        distill_mode = str(getattr(self, "distillation_mode", "kl"))
        if distill_mode == "mse":
            distill_loss = F.mse_loss(out_panel["qz_m"], out_full["qz_m"].detach())
        else:  # "kl"
            panel_qz_v = out_panel["qz_v"].clamp(min=1e-4, max=1e4)
            full_qz_v = out_full["qz_v"].clamp(min=1e-4, max=1e4)
            distill_loss = (
                kl(
                    Normal(out_panel["qz_m"], panel_qz_v.sqrt()),
                    Normal(out_full["qz_m"].detach(), full_qz_v.sqrt().detach()),
                )
                .sum(dim=1)
                .mean()
            )

        # Learnable temperature: stored as log(T), clamped to [0.01, 1.0]
        snn_temp = self.snn_temperature.exp().clamp(0.01, 1.0)
        snn = soft_nn_loss(out_panel["qz_m"], out_full["qz_m"].detach(), temperature=snn_temp)

        inv_loss = w.get("distill", 1.0) * distill_loss + w.get("snn", 0.1) * snn

        kl_w = self._kl_annealing_weight() * w["kl_weight"]
        total = (
            w["recon"] * reconst_loss
            + kl_w * kl_loss
            + kl_l.mean()  # library KL always weight-1
            + w["inv"] * inv_loss
            + lambda_pearson * pearson_loss
        )

        return {
            "loss": total,
            "reconst_loss": reconst_loss,
            "kl_loss": kl_loss,
            "kl_l_loss": kl_l.mean(),
            "distill_loss": distill_loss,
            "snn_loss": snn,
            "inv_loss": inv_loss,
            "snn_temperature": snn_temp,
            "pearson_loss": pearson_loss,
        }

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a training step for the current stage."""
        return self._shared_step(batch, batch_idx, prefix="train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a validation step for the current stage."""
        return self._shared_step(batch, batch_idx, prefix="val")

    def _shared_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        prefix: str,
    ) -> torch.Tensor:
        if self._training_stage == "pretrain":
            losses = self.compute_pretrain_losses(batch)
        else:
            losses = self.compute_losses(batch)

        log_kw: dict[str, Any] = {
            "prog_bar": True,
            "on_step": False,
            "on_epoch": True,
        }
        if prefix == "val":
            log_kw["sync_dist"] = True

        for name, val in losses.items():
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                self.log(f"{prefix}_{name}", val, **log_kw)

        return losses["loss"]

    # Optimiser

    def configure_optimizers(self):
        """AdamW optimiser with cosine-annealing LR schedule."""
        include_all_for_warm_unfreeze = (
            self._training_stage == "main" and self._decoder_warm_unfreeze_epoch >= 0 and not self._decoder_unfrozen
        )
        if include_all_for_warm_unfreeze:
            params = list(self.parameters())
        else:
            params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(self.hparams.get("lr", 1e-3)),
            weight_decay=float(self.hparams.get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.hparams.get("max_epochs", 100)),
            eta_min=float(self.hparams.get("lr", 1e-3)) * 0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }

    # ------------------------------------------------------------------
    # High-level training API
    # ------------------------------------------------------------------

    def pretrain_model(
        self,
        dataset: scAnnDataset,
        custom_callbacks: list | None = None,
        train_size: float = 0.8,
        pretrain_epochs: int = 50,
        **trainer_kwargs,
    ):
        """Stage-1 pretraining (full-gene view only, ELBO).

        Args:
            dataset: Training dataset.
            custom_callbacks: Extra PyTorch-Lightning callbacks.
            train_size: Fraction of data used for training.
            pretrain_epochs: Default max epochs (overridden by
                ``trainer_kwargs['max_epochs']`` if present).
            **trainer_kwargs: Forwarded to :class:`~cellpin.training.CellPinTrainer`.

        Returns:
        -------
            Fitted :class:`~cellpin.training.CellPinTrainer`.
        """
        self._training_stage = "pretrain"
        self._freeze_pretrained_in_main = False
        self._decoder_warm_unfreeze_epoch = -1
        self._decoder_unfrozen = True
        self._set_stage_trainability("pretrain", freeze_pretrained=False)

        max_epochs = trainer_kwargs.get("max_epochs", pretrain_epochs)
        self.hparams["max_epochs"] = max_epochs

        train_loader, val_loader = build_data_loaders(
            dataset,
            train_size=train_size,
            batch_size=trainer_kwargs.pop("batch_size", 128),
            num_workers=trainer_kwargs.pop("num_workers", 4),
        )
        trainer_kwargs = {**trainer_kwargs, "max_epochs": max_epochs}
        trainer = CellPinTrainer(custom_callbacks=custom_callbacks, **trainer_kwargs)
        try:
            trainer.fit(self, train_loader, val_loader)
        finally:
            self._training_stage = "main"
            self._pretrain_completed = True
        self._pretrain_output_dir = Path(trainer.logger[1].log_dir)

        best = trainer.best_model_path
        if best:
            ckpt = torch.load(best, map_location="cpu")
            self.load_state_dict(ckpt["state_dict"])
            print(f"  [pretrain] Restored best checkpoint (epoch {ckpt.get('epoch', '?')}): {Path(best).name}")

        return trainer

    def train_model(
        self,
        dataset: scAnnDataset,
        custom_callbacks: list | None = None,
        train_size: float = 0.8,
        freeze_pretrained: bool = False,
        require_pretrained: bool = True,
        **trainer_kwargs,
    ):
        """Stage-2 main training (both views, full ELBO + invariance + SNN).

        Args:
            dataset: Training dataset (:class:`~cellpin.dataset.scAnnDataset`).
            custom_callbacks: Extra PyTorch-Lightning callbacks.
            train_size: Fraction of data used for training.
            freeze_pretrained: If ``True``, freeze the full-gene encoder and
                decoder (Stage 1 weights) during Stage 2.
            require_pretrained: If ``True`` (default), raise an error when
                ``freeze_pretrained=True`` but ``pretrain_model`` was never
                called, preventing silent training against a random frozen
                decoder.
            **trainer_kwargs: Forwarded to :class:`~cellpin.training.CellPinTrainer`.

        Returns:
        -------
            Fitted :class:`~cellpin.training.CellPinTrainer`.

        Raises:
        ------
            RuntimeError: If ``require_pretrained=True``, ``freeze_pretrained=True``,
                and pretraining has not been completed.
        """
        if require_pretrained and freeze_pretrained and not self._pretrain_completed:
            raise RuntimeError(
                "freeze_pretrained=True but pretrain_model() has not been called. "
                "Run pretrain_model() first, or pass freeze_pretrained=False / "
                "require_pretrained=False to skip this check."
            )

        self._training_stage = "main"
        self._freeze_pretrained_in_main = freeze_pretrained

        warm_unfreeze_epoch = int(
            trainer_kwargs.pop(
                "decoder_warm_unfreeze_epoch",
                getattr(self, "decoder_warm_unfreeze_epoch", -1),
            )
        )
        self._decoder_warm_unfreeze_epoch = (
            warm_unfreeze_epoch if freeze_pretrained and warm_unfreeze_epoch >= 0 else -1
        )

        decoder_initial_trainable = self._decoder_warm_unfreeze_epoch < 0
        self._decoder_unfrozen = decoder_initial_trainable
        self._set_stage_trainability(
            "main",
            freeze_pretrained=freeze_pretrained,
            decoder_trainable_override=decoder_initial_trainable,
        )
        self.hparams["max_epochs"] = trainer_kwargs.get("max_epochs", 100)

        train_loader, val_loader = build_data_loaders(
            dataset,
            train_size=train_size,
            batch_size=trainer_kwargs.pop("batch_size", 128),
            num_workers=trainer_kwargs.pop("num_workers", 4),
        )
        trainer = CellPinTrainer(custom_callbacks=custom_callbacks, **trainer_kwargs)
        trainer.fit(self, train_loader, val_loader)
        self._train_output_dir = Path(trainer.logger[1].log_dir)

        best = trainer.best_model_path
        if best:
            ckpt = torch.load(best, map_location="cpu")
            self.load_state_dict(ckpt["state_dict"])
            print(f"  [train] Restored best checkpoint (epoch {ckpt.get('epoch', '?')}): {Path(best).name}")

        return trainer

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    @staticmethod
    def _panel_from_batch(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "panel_expr" in batch:
            return batch["panel_expr"]
        if "full_expr" in batch:
            return batch["full_expr"]
        raise KeyError("Batch must contain 'panel_expr' or 'full_expr'.")

    @torch.no_grad()
    def get_cell_embedding(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_mean: bool = True,
    ) -> np.ndarray:
        """Encode cells to the latent space via the panel encoder.

        Args:
            dataloader: DataLoader over a
                :class:`~cellpin.dataset.scAnnDataset` or
                :class:`~cellpin.dataset.stAnnDataset`.
            use_mean: Return the posterior mean rather than a sample.

        Returns:
        -------
            Float32 array ``(n_cells, n_latent)``.
        """
        self.eval()
        encoder = self.vae.z_encoder_panel
        embs = []
        for batch in track(dataloader, description="Embedding cells"):
            x_panel = self._panel_from_batch(batch).to(self.device)
            if self.vae.log_variational:
                x_panel = torch.log1p(x_panel)
            qz_m, _, z = encoder(x_panel)
            embs.append((qz_m if use_mean else z).cpu())
        return torch.cat(embs, dim=0).numpy()

    @torch.no_grad()
    def embed_and_impute(
        self,
        dataloader,
        use_mean: bool = True,
        mc_impute: bool = False,
        mc_samples: int = 50,
        mask_fraction: float = 0.2,
    ):
        """Embed cells and generate imputed expression values."""
        self.eval()  # required: DecoderSCVI uses BatchNorm1d (must use running stats)
        if torch.cuda.is_available() and self.device.type == "cpu":
            self.cuda()
        if mc_impute:
            print(f"Embedding and imputing cells (MC, {mc_samples} samples)...")
        else:
            print("Embedding and imputing cells...")

        # --- Panel gene order safeguard ---
        # Detect if the dataset's panel gene order differs from what the model was
        # trained with (sc_adata positional order) and compute a permutation to fix it.
        panel_perm: torch.Tensor | None = None
        dataset = dataloader.dataset
        if hasattr(dataset, "panel_genes") and hasattr(self, "panel_gene_names"):
            ds_panel = list(dataset.panel_genes)
            model_panel = self.panel_gene_names
            if ds_panel != model_panel:
                ds_idx = {g: i for i, g in enumerate(ds_panel)}
                missing = [g for g in model_panel if g not in ds_idx]
                if missing:
                    raise ValueError(f"Dataset is missing {len(missing)} genes the model expects: {missing[:10]} ...")
                perm = [ds_idx[g] for g in model_panel]
                panel_perm = torch.tensor(perm, dtype=torch.long)
                print(
                    f"[CellPin.impute] Panel gene order mismatch detected — "
                    f"auto-reordering {len(perm)} genes to match model's sc_adata order."
                )
            else:
                print("[CellPin.impute] Panel gene order confirmed ✓")

        embs, px_rates, libraries = [], [], []

        z_encoder = self.vae.z_encoder_panel

        for batch in dataloader:
            x_panel = self._panel_from_batch(batch).to(self.device)
            if panel_perm is not None:
                x_panel = x_panel[:, panel_perm.to(x_panel.device)]
            batch_index = batch.get("batch_index", None)
            if batch_index is not None:
                batch_index = batch_index.to(self.device)
            elif self.vae.n_batch > 0:
                batch_index = self.vae._get_infer_batch_cat(x_panel.size(0), self.device)

            x_panel_ = torch.log1p(x_panel) if self.vae.log_variational else x_panel

            # deterministic encodings
            qz_m, _, _ = z_encoder(x_panel_)
            ql_m, _, _ = self.vae.l_encoder(x_panel_)

            if mc_impute:
                samples = []
                for _ in range(mc_samples):
                    x_in = x_panel

                    if mask_fraction > 0.0:
                        keep = torch.bernoulli(torch.full_like(x_in, 1.0 - mask_fraction))
                        x_in = x_in * keep

                    x_in_ = torch.log1p(x_in) if self.vae.log_variational else x_in

                    _, _, z_s = z_encoder(x_in_)

                    if self.vae.reconstruction_loss in {"normal", "zin"}:
                        px_rate_s, _, _ = self.vae.decoder(z_s, batch_index)
                    else:
                        _, _, px_rate_s, _ = self.vae.decoder(self.vae.dispersion, z_s, ql_m, batch_index)

                    samples.append(px_rate_s)

                stacked = torch.stack(samples, dim=0)  # [mc_samples, n_cells, n_genes]
                px_rate = stacked.mean(dim=0)

            else:
                library_use = ql_m if use_mean else self.vae.l_encoder(x_panel_)[2]

                if self.vae.reconstruction_loss in {"normal", "zin"}:
                    px_rate, _, _ = self.vae.decoder(qz_m, batch_index)
                else:
                    _, _, px_rate, _ = self.vae.decoder(self.vae.dispersion, qz_m, library_use, batch_index)

            # keep observed panel genes if needed
            if not self._reconstruct_panel and self.vae.panel_idx is not None:
                px_rate = px_rate.clone()
                px_rate[:, self.vae.panel_idx] = x_panel

            embs.append(qz_m.cpu())
            px_rates.append(px_rate.cpu())
            libraries.append(ql_m.cpu())  # log-library

        return (
            torch.cat(embs).numpy(),
            torch.cat(px_rates).numpy(),
            torch.cat(libraries).numpy(),  # log-library
        )

    def _build_output_anndata(
        self,
        counts: np.ndarray,
        embeddings: np.ndarray,
        obs_adata: ad.AnnData | None,
        return_sparse: bool = True,
    ) -> ad.AnnData:
        """Build the base output AnnData for impute().

        Sets var_names, X_cellpin embedding, and copies obs/obsm/layers from obs_adata.
        Genes absent from obs_adata layers are filled with 0; var['is_measured'] marks
        which genes were present in obs_adata.
        """
        import scipy.sparse as sp

        adata_out = ad.AnnData(X=counts)
        adata_out.var_names = self.gene_names
        adata_out.obsm["X_cellpin"] = embeddings

        if obs_adata is not None:
            if obs_adata.n_obs != adata_out.n_obs:
                raise ValueError(f"obs_adata has {obs_adata.n_obs} cells; imputation has {adata_out.n_obs}.")
            adata_out.obs = obs_adata.obs.copy()

            for key, val in obs_adata.obsm.items():
                adata_out.obsm[key] = np.asarray(val)

            if obs_adata.n_vars > 0 and len(obs_adata.layers) > 0:
                out_gene_idx = {g: i for i, g in enumerate(adata_out.var_names)}
                src_cols = [out_gene_idx[g] for g in obs_adata.var_names if g in out_gene_idx]
                src_mask = [g in out_gene_idx for g in obs_adata.var_names]
                n_missing = adata_out.n_vars - len(src_cols)
                if n_missing > 0:
                    print(f"  [impute] Filling {n_missing} gene(s) absent from obs_adata layers with 0")
                for lyr_key, lyr_val in obs_adata.layers.items():
                    if hasattr(lyr_val, "toarray"):
                        lyr_val = lyr_val.toarray()
                    mat = np.zeros((adata_out.n_obs, adata_out.n_vars), dtype=np.float32)
                    mat[:, src_cols] = np.asarray(lyr_val, dtype=np.float32)[:, src_mask]
                    adata_out.layers[lyr_key] = sp.csr_matrix(mat) if return_sparse else mat

            # Mark which genes were measured in obs_adata
            measured = np.zeros(adata_out.n_vars, dtype=bool)
            out_gene_set = set(adata_out.var_names)
            for i, g in enumerate(adata_out.var_names):
                measured[i] = g in set(obs_adata.var_names) and g in out_gene_set
            adata_out.var["is_measured"] = measured
        else:
            # No obs_adata: all output genes are considered measured (imputed from sc ref)
            adata_out.var["is_measured"] = np.ones(adata_out.n_vars, dtype=bool)

        return adata_out

    def fit(
        self,
        dataset: scAnnDataset,
        pretrain_epochs: int = 50,
        train_epochs: int = 60,
        batch_size: int = 256,
        gradient_clip_val: float = 0.5,
        early_stopping_patience: int = 10,
        freeze_pretrained: bool = False,
        train_size: float = 0.8,
        save_checkpoints: bool = False,
        output_dir: str = "./cellpin_output",
        decoder_warm_unfreeze_epoch: int = -1,
        **trainer_kwargs,
    ) -> None:
        """Train CellPin: Stage 1 (pretrain) followed by Stage 2 (distillation).

        This is the recommended entry point for training. It runs both stages
        sequentially with a single call.

        Args:
            dataset: Single-cell dataset returned by :func:`cellpin.pp.setup`.
            pretrain_epochs: Max epochs for Stage 1 (full-gene ELBO pretraining).
            train_epochs: Max epochs for Stage 2 (panel distillation).
            batch_size: Mini-batch size for both stages.
            gradient_clip_val: Gradient clipping value.
            early_stopping_patience: Epochs without improvement before stopping.
            freeze_pretrained: Freeze the full-gene encoder/decoder during Stage 2.
            train_size: Fraction of cells used for training (rest → validation).
            save_checkpoints: Save model checkpoints to ``output_dir``.
                Disabled by default — enable when you need to resume training
                or load the best epoch after early stopping.
            output_dir: Root directory for checkpoints and logs
                (only used when ``save_checkpoints=True``).
            decoder_warm_unfreeze_epoch: Stage 2 epoch at which the frozen
                decoder is unfrozen for warm fine-tuning. Only active when
                ``freeze_pretrained=True``. ``-1`` (default) keeps the decoder
                frozen for the entire Stage 2 run.
            **trainer_kwargs: Extra arguments forwarded to
                :class:`~cellpin.training.CellPinTrainer` (e.g. ``devices``,
                ``precision``, ``accelerator``).

        Example::

            sc_dataset, _ = cellpin.pp.setup_data(sc_adata, st_adata)
            model = cellpin.CellPin(sc_dataset)
            model.fit(sc_dataset)
        """
        shared = dict(
            batch_size=batch_size,
            gradient_clip_val=gradient_clip_val,
            early_stopping_patience=early_stopping_patience,
            train_size=train_size,
            enable_checkpointing=save_checkpoints,
            **trainer_kwargs,
        )
        self.pretrain_model(
            dataset=dataset,
            pretrain_epochs=pretrain_epochs,
            output_dir=f"{output_dir}/pretrain",
            **shared,
        )
        self.train_model(
            dataset=dataset,
            freeze_pretrained=freeze_pretrained,
            max_epochs=train_epochs,
            output_dir=f"{output_dir}/train",
            decoder_warm_unfreeze_epoch=decoder_warm_unfreeze_epoch,
            **shared,
        )

    @torch.no_grad()
    def impute(
        self,
        dataloader: torch.utils.data.DataLoader,
        obs_adata: ad.AnnData | None = None,
        mc_samples: int = 50,
        mask_fraction: float = 0.2,
        return_norm: bool = False,
        norm_target_sum: float = 1e3,
        area_key: str | None = None,
        nb_count_samples: int = 100,
        nb_seed: int | None = None,
        return_int: bool = False,
        return_sparse: bool = True,
        table_key: str = "table",
    ) -> ad.AnnData:
        """Impute with MC averaging and optional count-space normalisation.

        Args:
            dataloader: DataLoader to run inference on.
            obs_adata: Optional AnnData (or :class:`spatialdata.SpatialData`) whose
                ``.obs`` is copied to the output. If SpatialData, the AnnData is read
                from ``obs_adata.tables[table_key]`` and the result is returned as an
                updated SpatialData object.  Must have the same number of observations.
            mc_samples: Number of stochastic forward passes for MC averaging
                (default 50; more → smoother but slower).
            mask_fraction: Fraction of panel genes randomly zeroed per MC pass
                to simulate missing measurements (default 0.2).
            return_norm: If ``True``, add a log-normalised layer
                ``layers['imputed_norm']`` (total-count or area normalised,
                then log1p-transformed).
            norm_target_sum: Target total counts for normalisation
                (default 1e3; only used when ``return_norm=True``).
            area_key: ``obs`` column with cell area for area-based normalisation.
                Auto-detected as ``'cell_area'`` when present; pass ``None``
                for total-count normalisation (only used when ``return_norm=True``).
            nb_count_samples: Number of NB draws used to compute the MC estimate
                of ``E[log1p(norm(X))]`` when ``return_norm=True`` (default 100).
                Because log1p is concave, Jensen's inequality means
                ``log1p(norm(E[X])) > E[log1p(norm(X))]``; sampling inside the
                transform corrects this bias.  More samples → lower variance.
            nb_seed: Seed for the NB draws (only used when ``return_norm=True``).
                Pass an integer for a reproducible ``layers['imputed_norm']``;
                ``None`` (default) draws fresh entropy on every call.  For a
                fixed seed the result is independent of thread count, though
                CPU and GPU use different samplers and so agree only up to
                Monte-Carlo error.
            return_int: If ``True``, round ``X`` to integer counts (``int32``).
            return_sparse: If ``True`` (default), store ``X``, ``layers['imputed']``,
                and ``layers['imputed_norm']`` as :class:`scipy.sparse.csr_matrix`.
                Set to ``False`` to keep dense numpy arrays.
            table_key: Table name to read/write when ``obs_adata`` is a SpatialData
                object (default ``"table"``).

        Returns:
        -------
            :class:`anndata.AnnData` with ``X`` = imputed (float or int) counts,
            ``obsm['X_cellpin']`` = embeddings, ``layers['imputed']`` = copy of
            ``X``, and optionally ``layers['imputed_norm']``.
            ``var['is_measured']`` marks genes present in ``obs_adata`` (all ``True``
            when ``obs_adata`` is ``None``).
            If ``obs_adata`` was a SpatialData object, returns the updated SpatialData
            with the result stored in ``sdata.tables[table_key]``.

        Raises:
        ------
            ValueError: If ``obs_adata`` has the wrong number of cells, or if
                ``area_key`` is specified but not found in ``adata.obs``, or if
                any cell area is ≤ 0.
        """
        import scipy.sparse as sp

        obs_adata, sdata = _resolve_sdata(obs_adata, table_key)

        embeddings, counts, log_library = self.embed_and_impute(
            dataloader,
            use_mean=True,
            mc_impute=True,
            mc_samples=mc_samples,
            mask_fraction=mask_fraction,
        )

        # Raw counts: smooth MC-averaged px_rate is the best point estimate.
        # Simple round preserves signal for count-level correlation.
        if return_int:
            counts = np.round(counts, decimals=0).astype(np.int32)

        adata_out = self._build_output_anndata(counts, embeddings, obs_adata, return_sparse=return_sparse)

        imputed = sp.csr_matrix(counts) if return_sparse else counts.copy()
        adata_out.X = imputed
        adata_out.layers["imputed"] = imputed

        if return_norm:
            # MC estimate of E[log1p(norm(X))] where X ~ NB(mu, theta).
            # Because log1p is concave, Jensen's inequality means
            # log1p(norm(E[X])) > E[log1p(norm(X))].  We correct this by
            # drawing K samples from NB, normalising and log1p-ing each draw,
            # then averaging — so norm+log1p go *inside* the MC loop.
            px_r_np = np.exp(self.vae.px_r.detach().cpu().numpy()).astype(np.float64)  # (n_genes,)

            # Resolve area column: explicit > auto-detect "cell_area" > None
            resolved_area_key = area_key
            if resolved_area_key is None and "cell_area" in adata_out.obs.columns:
                resolved_area_key = "cell_area"

            if resolved_area_key is not None:
                if resolved_area_key not in adata_out.obs.columns:
                    raise ValueError(f"area_key='{resolved_area_key}' not found in adata.obs")
                cell_area = adata_out.obs[resolved_area_key].values.astype(np.float64)
                if np.any(cell_area <= 0):
                    raise ValueError("All cell areas must be positive.")
                scale = norm_target_sum / cell_area  # (cells,)
            else:
                scale = None  # use per-draw library size

            norm_layer = mc_log1p_norm(
                counts,
                px_r_np,
                n_samples=max(nb_count_samples, 1),
                norm_target_sum=norm_target_sum,
                scale=scale,
                seed=nb_seed,
                device=self.device,
            )
            adata_out.layers["imputed_norm"] = sp.csr_matrix(norm_layer) if return_sparse else norm_layer

        if sdata is not None:
            sdata.tables[table_key] = adata_out
            return sdata
        return adata_out
