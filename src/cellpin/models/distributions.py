"""Probability distributions used in CellPin VAE models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, constraints
from torch.distributions import Normal as _TorchNormal
from torch.distributions import Poisson as _TorchPoisson
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
)


def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-probability of NegativeBinomial in terms of mean and dispersion.

    Args:
        x: Observed counts tensor.
        mu: Mean of the NegativeBinomial.
        theta: Inverse dispersion (theta -> inf gives Poisson).
        eps: Small constant for numerical stability.

    Returns:
    -------
        Log-probability per element.
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)

    res = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    return res


def log_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    zi_logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-probability of Zero-Inflated NegativeBinomial.

    Args:
        x: Observed counts tensor.
        mu: Mean of the NegativeBinomial component.
        theta: Inverse dispersion.
        zi_logits: Logits for the zero-inflation Bernoulli.
        eps: Small constant for numerical stability.

    Returns:
    -------
        Log-probability per element.
    """
    softplus_pi = F.softplus(-zi_logits)

    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)

    pi_theta_log = -zi_logits + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero
    return res


class NegativeBinomial(Distribution):
    """NegativeBinomial distribution parameterised by mean and inverse dispersion.

    Args:
        mu: Mean of the distribution (>0).
        theta: Inverse dispersion parameter (>0). As theta → ∞ the distribution
            converges to a Poisson.
        validate_args: Whether to validate args.
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        validate_args: bool = False,
    ):
        self.mu, self.theta = broadcast_all(mu, theta)
        super().__init__(validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return elementwise log-probability for observed counts."""
        return log_nb_positive(value, mu=self.mu, theta=self.theta)

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        return self.mu

    @lazy_property
    def variance(self) -> torch.Tensor:
        """Return the variance of the distribution."""
        return self.mu + self.mu.pow(2) / self.theta


class ZeroInflatedNegativeBinomial(NegativeBinomial):
    """Zero-inflated NegativeBinomial distribution.

    Args:
        mu: Mean of the NegativeBinomial component.
        theta: Inverse dispersion parameter.
        zi_logits: Logits for the zero-inflation probability.
        validate_args: Whether to validate args.
    """

    arg_constraints = {
        "mu": constraints.greater_than_eq(0),
        "theta": constraints.greater_than_eq(0),
        "zi_logits": constraints.real,
    }

    def __init__(
        self,
        mu: torch.Tensor,
        theta: torch.Tensor,
        zi_logits: torch.Tensor,
        validate_args: bool = False,
    ):
        super().__init__(mu, theta, validate_args=validate_args)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return elementwise log-probability for observed counts."""
        return log_zinb_positive(value, mu=self.mu, theta=self.theta, zi_logits=self.zi_logits)

    @lazy_property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        pi = torch.sigmoid(self.zi_logits)
        return (1 - pi) * self.mu


def log_zin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    zi_logits: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Log-probability of Zero-Inflated Normal.

    Models the mixture:  π·δ(x=0) + (1−π)·N(x; μ, σ²)

    For x = 0:  log[ π + (1−π)·N(0; μ, σ) ]
    For x ≠ 0:  log(1−π) + log N(x; μ, σ)

    Args:
        x: Observed log1p-normalised expression ``(batch, genes)``.
        mu: Predicted mean of the Normal component.
        sigma2: Predicted variance (> 0) of the Normal component.
        zi_logits: Zero-inflation logits (log π/(1−π)).
        eps: Numerical stability floor for σ.

    Returns:
    -------
        Log-probability per element, same shape as ``x``.
    """
    sigma = (sigma2 + eps).sqrt()
    log_prob_normal = _TorchNormal(mu, sigma).log_prob(x)
    log_prob_normal_at_zero = _TorchNormal(mu, sigma).log_prob(torch.zeros_like(x))

    log_pi = -F.softplus(-zi_logits)  # log σ(zi_logits)
    log_1mpi = -F.softplus(zi_logits)  # log(1 − σ(zi_logits))

    case_zero = torch.logaddexp(log_pi, log_1mpi + log_prob_normal_at_zero)
    case_nonzero = log_1mpi + log_prob_normal

    zero_mask = (x.abs() < eps).float()
    return zero_mask * case_zero + (1.0 - zero_mask) * case_nonzero


class ZeroInflatedNormal(Distribution):
    """Zero-Inflated Normal distribution for log1p-normalised expression.

    Models excess zeros with a Bernoulli spike, and non-zero expression
    with a Normal component.

    Args:
        mu: Mean of the Normal component.
        sigma2: Variance of the Normal component (> 0).
        zi_logits: Logits for the zero-inflation Bernoulli.
        validate_args: Whether to validate distribution arguments.
    """

    arg_constraints = {
        "mu": constraints.real,
        "sigma2": constraints.positive,
        "zi_logits": constraints.real,
    }
    support = constraints.nonnegative

    def __init__(
        self,
        mu: torch.Tensor,
        sigma2: torch.Tensor,
        zi_logits: torch.Tensor,
        validate_args: bool = False,
    ):
        self.mu, self.sigma2, self.zi_logits = broadcast_all(mu, sigma2, zi_logits)
        super().__init__(validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return elementwise log-probability for observed values."""
        return log_zin(value, mu=self.mu, sigma2=self.sigma2, zi_logits=self.zi_logits)

    @lazy_property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        pi = torch.sigmoid(self.zi_logits)
        return (1.0 - pi) * self.mu


class Poisson(Distribution):
    """Thin wrapper around :class:`torch.distributions.Poisson`.

    Accepts ``rate`` as a positional argument to match the NB/ZINB interface.

    Args:
        rate: Rate (= mean) of the Poisson distribution.
        validate_args: Whether to validate args.
    """

    arg_constraints = {"rate": constraints.greater_than_eq(0)}
    support = constraints.nonnegative_integer

    def __init__(self, rate: torch.Tensor, validate_args: bool = False):
        self._poisson = _TorchPoisson(rate=rate, validate_args=validate_args)
        self.rate = rate
        super().__init__(validate_args=validate_args)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return elementwise log-probability for observed counts."""
        return self._poisson.log_prob(value)

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        return self.rate
