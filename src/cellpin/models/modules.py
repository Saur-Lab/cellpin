"""Neural network building blocks for CellPin VAE models.

Public API
----------
Encoders:
    ResidualMLPBlock, GEGLU, DropPath
    ViewMLPEncoder     – maps a view vector -> embedding (used inside VAE encoder)
    Encoder            – wraps ViewMLPEncoder, adds mean/var/sample heads

Decoders:
    FCLayers           – flexible FC stack (used inside decoders)
    DecoderSCVI        – deep NB/ZINB decoder

Utilities:
    one_hot
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ===========================================================================
# Utilities
# ===========================================================================


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One-hot encode an integer index tensor.

    Args:
        index: Integer tensor of shape ``(batch,)``.
        n_cat: Number of categories.

    Returns:
    -------
        Float tensor of shape ``(batch, n_cat)``.
    """
    onehot = torch.zeros(index.size(0), n_cat, device=index.device, dtype=torch.float32)
    onehot.scatter_(1, index.long().view(-1, 1), 1)
    return onehot


def _resolve_cat(cat: torch.Tensor, n_cat: int) -> torch.Tensor:
    """Return a float covariate vector for injection into FCLayers.

    If ``cat`` is already a float tensor (e.g. a soft one-hot passed at spatial
    inference time), it is returned unchanged.  Otherwise it is treated as a
    long integer index and one-hot-encoded.
    """
    if cat.is_floating_point():
        return cat
    return one_hot(cat, n_cat)


# ===========================================================================
#  Encoder building blocks
# ===========================================================================


class DropPath(nn.Module):
    """Stochastic depth / DropPath (per-sample).

    Randomly drops entire samples in a batch during training, which acts as
    a structured regulariser for deep residual networks.

    Args:
        drop_prob: Probability of dropping a sample's residual branch.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth to the input tensor."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class GEGLU(nn.Module):
    """Gated GELU activation: splits projection into two halves, gates with GELU.

    Args:
        dim_in: Input dimensionality.
        dim_out: Output dimensionality (projection is ``dim_out * 2``).
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GEGLU activation to the input tensor."""
        a, b = self.proj(x).chunk(2, dim=-1)
        return a * self.act(b)


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual block with GEGLU FFN, LayerScale, and DropPath.

    Args:
        dim: Hidden dimensionality (input = output).
        expansion: GEGLU inner dimension multiplier.
        dropout: Dropout probability inside the FFN.
        drop_path: DropPath probability (stochastic depth).
        layer_scale_init: Initial value for LayerScale parameters.
    """

    def __init__(
        self,
        dim: int,
        expansion: float = 2.0,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        inner = int(dim * expansion)
        self.ff = nn.Sequential(
            GEGLU(dim, inner),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(inner, dim),
        )
        self.drop_path = DropPath(drop_path)
        self.layer_scale = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block forward pass."""
        h = self.ff(self.norm(x))
        return x + self.drop_path(h * self.layer_scale)


class ViewMLPEncoder(nn.Module):
    """Maps a single-view gene-expression vector to a dense embedding.

    Architecture: ``Linear → GELU → [ResidualMLPBlock] × num_blocks``

    Args:
        input_dim: Number of input genes (view-specific).
        embedding_dim: Output embedding dimensionality.
        num_blocks: Number of :class:`ResidualMLPBlock` layers.
        dropout: Dropout probability inside each block.
        drop_path_rate: Maximum DropPath rate; linearly increased across depth.
        ffn_expansion: GEGLU inner-dimension multiplier.
        layer_scale_init: Initial LayerScale value.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_blocks: int = 4,
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_expansion: float = 2.0,
        layer_scale_init: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Linearly increasing DropPath rates across depth (common practice)
        dp_rates = torch.linspace(0, drop_path_rate, steps=max(num_blocks, 1)).tolist()

        self.net = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            *[
                ResidualMLPBlock(
                    dim=embedding_dim,
                    expansion=ffn_expansion,
                    dropout=dropout,
                    drop_path=dp_rates[i],
                    layer_scale_init=layer_scale_init,
                )
                for i in range(num_blocks)
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single-view expression vector."""
        if x.size(-1) != self.input_dim:
            raise ValueError(f"ViewMLPEncoder: expected input_dim={self.input_dim}, got {x.size(-1)}")
        return self.net(x)


# ===========================================================================
# VAE Encoder
# ===========================================================================


class Encoder(nn.Module):
    """VAE encoder: ``ViewMLPEncoder`` body + mean/var/reparameterisation heads.

    Args:
        n_input: Input gene dimensionality (view-specific).
        n_output: Latent space dimensionality.
        n_hidden: Embedding dimensionality of the encoder body.
        n_layers: Number of :class:`ResidualMLPBlock` layers.
        dropout_rate: Dropout inside each residual block.
        drop_path_rate: Maximum DropPath rate (linearly scaled across depth).
        ffn_expansion: GEGLU FFN expansion factor.
        layer_scale_init: Initial LayerScale value.
        distribution: ``'normal'`` or ``'ln'`` (log-normal).
        input_noise_std: Standard deviation of Gaussian noise injected onto the
            input at the start of the forward pass, training only. 0.0 = disabled.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int = 128,
        n_layers: int = 4,
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_expansion: float = 2.0,
        layer_scale_init: float = 1e-3,
        distribution: str = "normal",
        input_noise_std: float = 0.0,
    ):
        super().__init__()
        self.distribution = distribution
        self.n_output = n_output
        self.input_noise_std = float(input_noise_std)

        self.encoder_body = ViewMLPEncoder(
            input_dim=n_input,
            embedding_dim=n_hidden,
            num_blocks=n_layers,
            dropout=dropout_rate,
            drop_path_rate=drop_path_rate,
            ffn_expansion=ffn_expansion,
            layer_scale_init=layer_scale_init,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

        if distribution == "ln":
            self.z_transformation: nn.Module = nn.Softmax(dim=-1)
        else:
            self.z_transformation = nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ``x`` to posterior parameters and a reparameterised sample.

        Args:
            x: Expression tensor ``(batch, n_input)``.

        Returns:
        -------
            ``(q_mean, q_var, z)`` — posterior mean, variance, and sample.
        """
        if self.training and self.input_noise_std > 0.0:
            x = x + torch.randn_like(x) * self.input_noise_std

        h = self.encoder_body(x)
        q_m = self.mean_encoder(h)
        # softplus instead of exp: grows linearly for large inputs (no fp16 overflow)
        q_v = F.softplus(self.var_encoder(h)) + 1e-4
        z = Normal(q_m, q_v.sqrt()).rsample()
        z = self.z_transformation(z)
        return q_m, q_v, z


# ===========================================================================
# Decoder building blocks
# ===========================================================================


class FCLayers(nn.Module):
    """Flexible fully-connected layer stack (scVI-style).

    Args:
        n_in: Input dimensionality.
        n_out: Output dimensionality.
        n_cat_list: Category counts for one-hot covariates to append.
        n_layers: Number of linear layers.
        n_hidden: Hidden layer width (unused for last layer, which is n_out).
        dropout_rate: Dropout probability.
        use_batch_norm: Apply BatchNorm1d after each linear layer.
        use_layer_norm: Apply LayerNorm after each linear layer.
        use_activation: Apply ReLU activation (not applied on the last layer).
        bias: Include bias in linear layers.
        inject_covariates: Inject covariates at every layer (not just input).
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        n_cat_list = list(n_cat_list) if n_cat_list is not None else []
        self.n_cat_list = [n if n > 1 else 0 for n in n_cat_list]
        cat_dim = sum(self.n_cat_list)

        layers_list = []
        for i in range(n_layers):
            in_dim = (n_in if i == 0 else n_hidden) + (cat_dim if inject_covariates else 0)
            out_dim = n_out if i == n_layers - 1 else n_hidden
            sub = []
            sub.append(nn.Linear(in_dim, out_dim, bias=bias))
            if use_batch_norm:
                sub.append(nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001))
            if use_layer_norm:
                sub.append(nn.LayerNorm(out_dim))
            if use_activation and i < n_layers - 1:
                sub.append(nn.ReLU())
            if dropout_rate > 0:
                sub.append(nn.Dropout(p=dropout_rate))
            layers_list.append(nn.Sequential(*sub))

        self.fc_layers = nn.Sequential(*layers_list)

    def inject_into_layer(self, layer_num: int) -> bool:
        """Return True if covariates should be injected at this layer."""
        return layer_num == 0 or self.inject_covariates

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor) -> torch.Tensor:
        """Forward pass through the FC stack with optional covariates."""
        one_hot_cats = []
        for n_cat, cat in zip(self.n_cat_list, cat_list, strict=False):
            if n_cat > 1:
                one_hot_cats.append(_resolve_cat(cat, n_cat))

        for i, layer in enumerate(self.fc_layers):
            if self.inject_into_layer(i) and one_hot_cats:
                x = torch.cat([x, *one_hot_cats], dim=-1)
            x = layer(x)
        return x


class DecoderSCVI(nn.Module):
    """scVI-style deep probabilistic decoder.

    Decodes ``(z, library, batch)`` to NB / ZINB parameters:
    ``(px_scale, px_r, px_rate, px_dropout)``.

    Args:
        n_input: Latent dimensionality.
        n_output: Number of genes.
        n_cat_list: Category sizes for one-hot covariates.
        n_layers: Number of hidden FC layers.
        n_hidden: Hidden layer width.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.0,
        )
        self.px_scale_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1))
        self.px_r_decoder = nn.Linear(n_hidden, n_output)
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """Decode latent to NB parameters.

        Args:
            dispersion: Dispersion mode; ``'gene-cell'`` uses ``px_r_decoder``.
            z: Latent tensor ``(batch, n_latent)``.
            library: Log-library tensor ``(batch, 1)``.
            *cat_list: Optional categorical covariates.

        Returns:
        -------
            ``(px_scale, px_r, px_rate, px_dropout)``
        """
        px = self.px_decoder(z, *cat_list)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        px_rate = torch.exp(library) * px_scale
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class DecoderNormal(nn.Module):
    """Decoder for log1p-normalised data (Normal / Zero-Inflated Normal).

    Outputs per-gene mean, variance, and zero-inflation logits directly
    from the latent vector — no library-size scaling.

    Args:
        n_input: Latent dimensionality.
        n_output: Number of genes.
        n_cat_list: Category sizes for one-hot covariates.
        n_layers: Number of hidden FC layers.
        n_hidden: Hidden layer width.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] | None = None,
        n_layers: int = 1,
        n_hidden: int = 128,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.0,
        )
        self.px_mu_decoder = nn.Linear(n_hidden, n_output)
        self.px_logvar_decoder = nn.Linear(n_hidden, n_output)
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        z: torch.Tensor,
        *cat_list: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode latent to Normal / ZIN parameters.

        Args:
            z: Latent tensor ``(batch, n_latent)``.
            *cat_list: Optional categorical covariates.

        Returns:
        -------
            ``(px_mu, px_sigma2, px_dropout)``:
            - ``px_mu``: predicted mean per gene.
            - ``px_sigma2``: predicted variance per gene (softplus of raw logvar).
            - ``px_dropout``: zero-inflation logits (unused for plain Normal).
        """
        px = self.px_decoder(z, *cat_list)
        px_mu = self.px_mu_decoder(px)
        px_sigma2 = F.softplus(self.px_logvar_decoder(px))
        px_dropout = self.px_dropout_decoder(px)
        return px_mu, px_sigma2, px_dropout
