# -*- coding: utf-8 -*-
"""
CellPin hybrid VAE models.

- Two-view training (full-gene + panel-gene paths).
- NB / ZINB / Poisson generative distributions.
- ``'gene'`` and ``'gene-cell'`` dispersion modes.
- Log(1+x) input stabilisation.
- Batch correction via one-hot categorical covariates.
"""

from __future__ import annotations

from typing import Dict, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

from cellpin.models.distributions import (
    NegativeBinomial,
    Poisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedNormal,
)
from cellpin.models.modules import (
    DecoderNormal,
    DecoderSCVI,
    Encoder,
)

_NORMAL_LOSSES: frozenset[str] = frozenset({"normal", "zin"})

torch.backends.cudnn.benchmark = True


# ===========================================================================
# CellPinVAE
# ===========================================================================

class CellPinVAE(nn.Module):
    """Hybrid two-view VAE for single-cell and spatial transcriptomics.

    Two encoder streams share the same latent space:

    * ``z_encoder_full`` infers reference latent geometry from full genes.
    * ``z_encoder_panel`` infers deployment latent geometry from panel genes.

    ``l_encoder`` infers log-library from panel genes.

    The decoder reconstructs full-gene expression.

    Args:
        n_input_full: Number of full-profile genes to reconstruct.
        n_input_panel: Number of panel genes used for inference.
        panel_idx: Optional index mapping panel genes in the full profile.
            Used as fallback if ``x_panel`` is not passed at inference.
        use_panel_only: If ``True`` (default), require panel input for
            library inference; latent view can be selected via ``encoder_view``.
        n_batch: Number of batches for batch correction (0 = off).
        n_hidden: Embedding width shared by encoders and decoder hidden layers.
        n_latent: Latent space dimensionality.
        n_layers_encoder: Number of ``ResidualMLPBlock`` layers per encoder.
        n_layers_decoder: Number of FC layers in the decoder.
        dropout_rate: Dropout inside residual blocks.
        drop_path_rate: Maximum DropPath rate (linearly scaled across depth).
        ffn_expansion: GEGLU FFN expansion factor in residual blocks.
        layer_scale_init: Initial value for LayerScale parameters.
        dispersion: One of:

            * ``'gene'``      — one θ per gene (global).
            * ``'gene-cell'`` — θ predicted per cell by decoder.

        log_variational: Apply log(1 + x) before encoding.
        reconstruction_loss: One of ``'nb'``, ``'zinb'``, ``'poisson'``, ``'normal'``, ``'zin'``.
        latent_distribution: ``'normal'`` or ``'ln'`` (log-normal).

    Example::

        model = CellPinVAE(
            n_input_full=2000,
            n_input_panel=450,
            n_latent=32,
            n_hidden=256,
            n_layers_encoder=6,
            reconstruction_loss="zinb",
        )
    """

    def __init__(
        self,
        n_input_full: int,
        n_input_panel: int,
        panel_idx: list[int] | np.ndarray | torch.Tensor | None = None,
        use_panel_only: bool = True,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 32,
        n_layers_encoder: int = 4,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_expansion: float = 2.0,
        layer_scale_init: float = 1e-3,
        dispersion: Literal["gene", "gene-cell"] = "gene",
        log_variational: bool = True,
        reconstruction_loss: Literal["nb", "zinb", "poisson", "normal", "zin"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        input_noise_std: float = 0.0,
        exclude_panel: bool = False,
        batch_infer_mode: str = "mean_onehot",
    ):
        super().__init__()
        self.n_input_full = n_input_full
        self.n_input_panel = n_input_panel
        self.n_batch = n_batch
        self.batch_infer_mode = batch_infer_mode
        self.n_latent = n_latent
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.reconstruction_loss = reconstruction_loss
        self.latent_distribution = latent_distribution
        self.use_panel_only = bool(use_panel_only)
        self.exclude_panel = bool(exclude_panel)
        if panel_idx is not None:
            panel_idx_tensor = torch.as_tensor(panel_idx, dtype=torch.long)
            self.register_buffer("panel_idx", panel_idx_tensor)
            # Registered as a buffer so it is automatically moved to the correct device.
            panel_mask_tensor = torch.zeros(n_input_full, dtype=torch.bool)
            panel_mask_tensor[panel_idx_tensor] = True
            self.register_buffer("panel_mask", panel_mask_tensor)
        else:
            self.panel_idx = None
            self.panel_mask = None

        # ------------------------------------------------------------------
        # Global dispersion parameters
        # ------------------------------------------------------------------
        if dispersion == "gene":
            self.px_r = nn.Parameter(torch.randn(n_input_full))
        elif dispersion == "gene-cell":
            pass  # predicted by decoder per cell
        else:
            raise ValueError(
                f"dispersion must be one of ['gene', 'gene-cell'], got '{dispersion}'."
            )

        # ------------------------------------------------------------------
        # Shared encoder kwargs (all residual MLP encoders use same settings)
        # ------------------------------------------------------------------
        enc_kwargs = dict(
            n_hidden=n_hidden,
            n_layers=n_layers_encoder,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            ffn_expansion=ffn_expansion,
            layer_scale_init=layer_scale_init,
            distribution=latent_distribution,
        )

        # View 1: full-gene encoder (reference geometry)
        self.z_encoder_full = Encoder(
            n_input=n_input_full,
            n_output=n_latent,
            input_noise_std=input_noise_std,
            **enc_kwargs,
        )
        # View 2: panel-gene encoder (deployment geometry)
        self.z_encoder_panel = Encoder(
            n_input=n_input_panel,
            n_output=n_latent,
            input_noise_std=input_noise_std,
            **enc_kwargs,
        )
        # Library-size encoder — shallow (1 layer), panel-only input.
        # [ADDED NOTE] Intentionally NO input_noise_std here: the library
        # encoder estimates total RNA content; noise would bias that estimate.
        self.l_encoder = Encoder(
            n_input=n_input_panel,
            n_output=1,
            n_hidden=n_hidden,
            n_layers=1,        # intentionally shallow
            dropout_rate=dropout_rate,
            distribution="normal",
            # input_noise_std intentionally omitted → defaults to 0.0
        )

        # ------------------------------------------------------------------
        # Decoder — scVI-style for count data, Normal-style for normalised data
        # ------------------------------------------------------------------
        if reconstruction_loss in _NORMAL_LOSSES:
            self.decoder = DecoderNormal(
                n_input=n_latent,
                n_output=n_input_full,
                n_cat_list=[n_batch],
                n_layers=n_layers_decoder,
                n_hidden=n_hidden,
            )
        else:
            self.decoder = DecoderSCVI(
                n_input=n_latent,
                n_output=n_input_full,
                n_cat_list=[n_batch],
                n_layers=n_layers_decoder,
                n_hidden=n_hidden,
            )

    # ------------------------------------------------------------------
    # Batch covariate helpers
    # ------------------------------------------------------------------

    def _get_batch_cat(self, batch_index: torch.Tensor | None) -> torch.Tensor | None:
        """Return the batch covariate tensor for the decoder (training path).

        For a long index tensor, FCLayers will call one_hot internally.
        Returns None when batch conditioning is off.
        """
        if batch_index is None or self.n_batch == 0:
            return None
        return batch_index  # long (B,) → FCLayers._resolve_cat → one_hot

    def _get_infer_batch_cat(self, n_cells: int, device: torch.device) -> torch.Tensor | None:
        """Return the batch covariate for spatial inference (no batch label known).

        ``batch_infer_mode='mean_onehot'`` (default): uniform soft one-hot of
        shape ``(n_cells, n_batch)``, each row = 1/n_batch.  Passed as a float
        tensor so FCLayers skips one_hot conversion.
        """
        if self.n_batch == 0:
            return None
        if self.batch_infer_mode == "mean_onehot":
            return torch.full(
                (n_cells, self.n_batch),
                1.0 / self.n_batch,
                device=device,
                dtype=torch.float32,
            )
        return None

    # ------------------------------------------------------------------
    # Posterior sampling helpers
    # ------------------------------------------------------------------

    def sample_from_posterior_z(
        self,
        x: torch.Tensor,
        view: Literal["full", "panel"] = "panel",
        give_mean: bool = False,
        n_samples: int = 5000,
    ) -> torch.Tensor:
        """Sample from q(z | x).

        Args:
            x: Expression tensor for the chosen view.
            view: Which encoder to use — ``'full'`` or ``'panel'``.
            give_mean: Return the posterior mean rather than a sample.
            n_samples: MC samples for log-normal mean approximation.

        Returns:
            Latent tensor ``(batch, n_latent)``.
        """
        if self.log_variational:
            x = torch.log1p(x)
        if view == "full" and self.exclude_panel and self.panel_mask is not None:
            x = x.masked_fill(self.panel_mask.unsqueeze(0), 0.0)
        if view == "full":
            encoder = self.z_encoder_full
        else:
            encoder = self.z_encoder_panel
        qz_m, qz_v, z = encoder(x)
        if give_mean:
            if self.latent_distribution == "ln":
                samples = Normal(qz_m, qz_v.sqrt()).sample([n_samples])
                z = encoder.z_transformation(samples).mean(dim=0)
            else:
                z = qz_m
        return z

    def sample_from_posterior_l(self, x: torch.Tensor) -> torch.Tensor:
        """Sample log-library size from q(l | x_panel).

        Args:
            x: Panel expression tensor ``(batch, n_input_panel)``.

        Returns:
            Library sample ``(batch, 1)``.
        """
        if self.log_variational:
            x = torch.log1p(x)
        _, _, library = self.l_encoder(x)
        return library

    # ------------------------------------------------------------------
    # Convenience output methods
    # ------------------------------------------------------------------

    def get_sample_scale(
        self,
        x: torch.Tensor,
        x_panel: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        n_samples: int = 1,
        transform_batch: int | None = None,
    ) -> torch.Tensor:
        """Predicted softmax gene-expression frequencies.

        Args:
            x: Full-gene expression ``(batch, n_input_full)``.
            x_panel: Panel expression (uses panel encoder if provided).
            batch_index: Integer batch labels.
            n_samples: Number of posterior samples to average.
            transform_batch: Decode under this batch instead of ``batch_index``.

        Returns:
            Scale tensor ``(batch, n_input_full)``.
        """
        return self.inference(
            x, x_panel=x_panel, batch_index=batch_index,
            n_samples=n_samples, transform_batch=transform_batch,
        )["px_scale"]

    def get_sample_rate(
        self,
        x: torch.Tensor,
        x_panel: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        n_samples: int = 1,
        transform_batch: int | None = None,
    ) -> torch.Tensor:
        """Predicted NB mean expression rates (= library × scale).

        Args:
            x: Full-gene expression ``(batch, n_input_full)``.
            x_panel: Panel expression (optional).
            batch_index: Integer batch labels.
            n_samples: Number of posterior samples.
            transform_batch: Decode under this batch instead of ``batch_index``.

        Returns:
            Rate tensor ``(batch, n_input_full)``.
        """
        return self.inference(
            x, x_panel=x_panel, batch_index=batch_index,
            n_samples=n_samples, transform_batch=transform_batch,
        )["px_rate"]

    # ------------------------------------------------------------------
    # Reconstruction loss
    # ------------------------------------------------------------------

    def get_reconstruction_loss(
        self,
        x: torch.Tensor,
        px_rate: torch.Tensor,
        px_r: torch.Tensor,
        px_dropout: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood under the chosen generative model.

        For count-based losses (``'nb'``, ``'zinb'``, ``'poisson'``):
            - ``px_rate``: NB mean (library-scaled).
            - ``px_r``: NB inverse dispersion.
            - ``px_dropout``: ZINB zero-inflation logits.

        For normalised-data losses (``'normal'``, ``'zin'``):
            - ``px_rate``: predicted mean μ.
            - ``px_r``: predicted variance σ² (softplus-transformed).
            - ``px_dropout``: ZIN zero-inflation logits (ignored for ``'normal'``).

        Returns:
            Per-cell NLL ``(batch,)``.
        """
        if self.reconstruction_loss == "zinb":
            per_gene = -ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout
            ).log_prob(x)
        elif self.reconstruction_loss == "nb":
            per_gene = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x)
        elif self.reconstruction_loss == "poisson":
            per_gene = -Poisson(px_rate).log_prob(x)
        elif self.reconstruction_loss == "normal":
            per_gene = -Normal(px_rate, (px_r + 1e-8).sqrt()).log_prob(x)
        elif self.reconstruction_loss == "zin":
            per_gene = -ZeroInflatedNormal(
                mu=px_rate, sigma2=px_r, zi_logits=px_dropout
            ).log_prob(x)
        else:
            raise ValueError(f"Unknown reconstruction_loss: '{self.reconstruction_loss}'")

        return per_gene.sum(dim=-1)

    # ------------------------------------------------------------------
    # Core inference pass
    # ------------------------------------------------------------------

    def inference(
        self,
        x: torch.Tensor,
        x_panel: torch.Tensor | None = None,
        encoder_view: Literal["panel", "full"] = "panel",
        batch_index: torch.Tensor | None = None,
        n_samples: int = 1,
        transform_batch: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run encoders and decoder; return all intermediate tensors.

        ``z`` is inferred from the selected encoder view, while library is
        always inferred from panel expression.

        Args:
            x: Full-gene expression ``(batch, n_input_full)``.
            x_panel: Panel expression ``(batch, n_input_panel)`` — required
                in panel-only mode, unless ``panel_idx`` is set.
            encoder_view: ``'panel'`` (default) or ``'full'`` for latent path.
            batch_index: Integer batch labels ``(batch,)``.
            n_samples: Draw multiple posterior samples (expand batch dim).
            transform_batch: Override batch index for the decoder.

        Returns:
            Dict with keys:
            ``px_scale``, ``px_r``, ``px_rate``, ``px_dropout``,
            ``qz_m``, ``qz_v``, ``z``,
            ``ql_m``, ``ql_v``, ``library``.
        """
        if x_panel is None:
            if self.panel_idx is not None:
                x_panel = x.index_select(1, self.panel_idx.to(x.device))
            elif self.use_panel_only:
                raise ValueError(
                    "Panel input required for panel-only mode. "
                    "Pass x_panel or provide panel_idx at init."
                )
            else:
                raise ValueError("x_panel is required.")

        x_panel_ = torch.log1p(x_panel) if self.log_variational else x_panel
        x_ = torch.log1p(x) if self.log_variational else x

        # ---- Encode z from selected view; library from panel ----
        if encoder_view == "full":
            if self.exclude_panel and self.panel_mask is not None:
                x_full_ = x_.masked_fill(self.panel_mask.unsqueeze(0), 0.0)
            else:
                x_full_ = x_  # default: full encoder sees all genes
            qz_m, qz_v, z = self.z_encoder_full(x_full_)
            z_encoder = self.z_encoder_full
        elif encoder_view == "panel":
            qz_m, qz_v, z = self.z_encoder_panel(x_panel_)
            z_encoder = self.z_encoder_panel
        else:
            raise ValueError(f"Unknown encoder_view='{encoder_view}'.")

        ql_m, ql_v, library = self.l_encoder(x_panel_)

        # ---- Multiple posterior samples ----
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand(n_samples, *qz_m.shape)
            qz_v = qz_v.unsqueeze(0).expand(n_samples, *qz_v.shape)
            z = Normal(qz_m, qz_v.sqrt()).rsample()
            z = z_encoder.z_transformation(z)
            ql_m = ql_m.unsqueeze(0).expand(n_samples, *ql_m.shape)
            ql_v = ql_v.unsqueeze(0).expand(n_samples, *ql_v.shape)
            library = Normal(ql_m, ql_v.sqrt()).rsample()

        # ---- Decode ----
        dec_batch = (
            transform_batch * torch.ones_like(batch_index)
            if transform_batch is not None
            else batch_index
        )
        # Spatial inference: no batch label available → use soft one-hot
        if dec_batch is None and self.n_batch > 0:
            _n = z.size(0) if z.ndim == 2 else z.size(1)
            dec_batch = self._get_infer_batch_cat(_n, z.device)

        if self.reconstruction_loss in _NORMAL_LOSSES:
            # DecoderNormal: no library scaling; returns (mu, sigma2, zi_logits)
            px_rate, px_r, px_dropout = self.decoder(z, dec_batch)
            px_scale = px_rate  # alias for API consistency; no softmax scale
        else:
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion, z, library, dec_batch
            )
            # ---- Resolve dispersion for count-based decoders ----
            if self.dispersion == "gene":
                px_r = self.px_r

            if self.dispersion != "gene-cell":
                px_r = torch.exp(px_r)

        return dict(
            px_scale=px_scale,
            px_r=px_r,
            px_rate=px_rate,
            px_dropout=px_dropout,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z,
            ql_m=ql_m,
            ql_v=ql_v,
            library=library,
        )

    # ------------------------------------------------------------------
    # Forward: ELBO components
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        local_l_mean: torch.Tensor,
        local_l_var: torch.Tensor,
        x_panel: torch.Tensor | None = None,
        encoder_view: Literal["panel", "full"] = "panel",
        batch_index: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute ELBO components for a mini-batch.

        Following the scVI convention the return tuple has three elements so
        the Lightning trainer can apply a separate KL weight:

        ``(reconst_loss + KL_l, KL_z, 0.0)``

        Args:
            x: Full-gene expression ``(batch, n_input_full)``.
            local_l_mean: Prior mean of log-library ``(batch, 1)``.
            local_l_var: Prior variance of log-library ``(batch, 1)``.
            x_panel: Panel expression used for library inference.
            encoder_view: ``'panel'`` or ``'full'`` for latent inference.
            batch_index: Integer batch labels.

        Returns:
            ``(reconst_loss + kl_l, kl_z, 0.0)``
        """
        outputs = self.inference(
            x,
            x_panel=x_panel,
            encoder_view=encoder_view,
            batch_index=batch_index,
        )
        qz_m, qz_v = outputs["qz_m"], outputs["qz_v"]
        ql_m, ql_v = outputs["ql_m"], outputs["ql_v"]

        # KL(q(z) || N(0,I))
        kl_z = kl(
            Normal(qz_m, qz_v.sqrt()),
            Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)),
        ).sum(dim=1)

        # KL(q(l) || p(l))  — data-driven prior from the dataset statistics
        kl_l = kl(
            Normal(ql_m, ql_v.sqrt()),
            Normal(local_l_mean, local_l_var.sqrt()),
        ).sum(dim=1)

        reconst = self.get_reconstruction_loss(
            x, outputs["px_rate"], outputs["px_r"], outputs["px_dropout"]
        )
        return reconst + kl_l, kl_z, 0.0
