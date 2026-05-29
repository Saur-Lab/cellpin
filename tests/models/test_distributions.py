import torch

from cellpin.models.distributions import (
    NegativeBinomial,
    Poisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedNormal,
    log_nb_positive,
    log_zinb_positive,
)

B, G = 4, 10


class TestLogNBPositive:
    def test_output_shape(self):
        x = torch.ones(B, G)
        mu = torch.full((B, G), 2.0)
        theta = torch.full((B, G), 3.0)
        assert log_nb_positive(x, mu, theta).shape == (B, G)

    def test_finite_for_large_x(self):
        x = torch.full((B, G), 100.0)
        mu = torch.ones(B, G)
        theta = torch.ones(B, G)
        assert torch.isfinite(log_nb_positive(x, mu, theta)).all()

    def test_finite_for_zero_counts(self):
        x = torch.zeros(B, G)
        mu = torch.ones(B, G)
        theta = torch.full((B, G), 2.0)
        assert torch.isfinite(log_nb_positive(x, mu, theta)).all()

    def test_non_positive_values(self):
        x = torch.ones(B, G) * 3
        mu = torch.ones(B, G)
        theta = torch.ones(B, G)
        assert (log_nb_positive(x, mu, theta) <= 0).all()


class TestLogZINBPositive:
    def test_output_shape(self):
        x = torch.ones(B, G)
        mu = torch.full((B, G), 2.0)
        theta = torch.full((B, G), 3.0)
        zi_logits = torch.zeros(B, G)
        assert log_zinb_positive(x, mu, theta, zi_logits).shape == (B, G)

    def test_finite(self):
        x = torch.rand(B, G) * 5
        mu = torch.rand(B, G).abs() + 0.1
        theta = torch.rand(B, G).abs() + 0.5
        zi_logits = torch.randn(B, G)
        assert torch.isfinite(log_zinb_positive(x, mu, theta, zi_logits)).all()

    def test_high_zi_logits_favour_zeros(self):
        x_zero = torch.zeros(B, G)
        x_nonzero = torch.ones(B, G) * 5
        mu = torch.ones(B, G)
        theta = torch.ones(B, G)
        zi_logits = torch.full((B, G), 10.0)  # pi ≈ 1 → zeros very likely
        lp_zero = log_zinb_positive(x_zero, mu, theta, zi_logits)
        lp_nonzero = log_zinb_positive(x_nonzero, mu, theta, zi_logits)
        assert (lp_zero > lp_nonzero).all()


class TestNegativeBinomial:
    def test_log_prob_shape(self):
        mu = torch.rand(B, G).abs() + 0.1
        theta = torch.rand(B, G).abs() + 0.5
        lp = NegativeBinomial(mu=mu, theta=theta).log_prob(torch.zeros(B, G))
        assert lp.shape == (B, G)

    def test_mean(self):
        mu = torch.tensor([1.0, 2.0, 3.0])
        theta = torch.ones(3)
        assert torch.allclose(NegativeBinomial(mu=mu, theta=theta).mean, mu)

    def test_variance_exceeds_mean(self):
        mu = torch.tensor([2.0])
        theta = torch.tensor([1.0])
        d = NegativeBinomial(mu=mu, theta=theta)
        assert d.variance > d.mean

    def test_log_prob_finite(self):
        mu = torch.rand(B, G).abs() + 0.1
        theta = torch.rand(B, G).abs() + 0.5
        x = (torch.rand(B, G) * 10).floor()
        assert torch.isfinite(NegativeBinomial(mu=mu, theta=theta).log_prob(x)).all()


class TestZeroInflatedNegativeBinomial:
    def test_log_prob_shape(self):
        mu = torch.rand(B, G).abs() + 0.1
        theta = torch.rand(B, G).abs() + 0.5
        zi_logits = torch.zeros(B, G)
        lp = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits).log_prob(torch.zeros(B, G))
        assert lp.shape == (B, G)

    def test_high_zi_mean_near_zero(self):
        mu = torch.tensor([5.0])
        theta = torch.tensor([1.0])
        zi_logits = torch.tensor([10.0])  # pi ≈ 1 → mean ≈ 0
        d = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
        assert d.mean < mu

    def test_low_zi_mean_near_nb(self):
        mu = torch.tensor([3.0])
        theta = torch.tensor([2.0])
        zi_logits = torch.tensor([-20.0])  # pi ≈ 0 → mean ≈ mu
        d = ZeroInflatedNegativeBinomial(mu=mu, theta=theta, zi_logits=zi_logits)
        assert torch.allclose(d.mean, mu, atol=1e-3)


class TestPoisson:
    def test_log_prob_shape(self):
        rate = torch.rand(B, G).abs() + 0.1
        lp = Poisson(rate=rate).log_prob(torch.ones(B, G))
        assert lp.shape == (B, G)

    def test_mean(self):
        rate = torch.tensor([1.5, 2.5])
        assert torch.allclose(Poisson(rate=rate).mean, rate)

    def test_log_prob_finite(self):
        rate = torch.rand(B, G).abs() + 0.1
        x = (torch.rand(B, G) * 5).floor()
        assert torch.isfinite(Poisson(rate=rate).log_prob(x)).all()


class TestZeroInflatedNormal:
    def test_log_prob_shape(self):
        mu = torch.randn(B, G)
        sigma2 = torch.rand(B, G).abs() + 0.1
        zi_logits = torch.zeros(B, G)
        lp = ZeroInflatedNormal(mu=mu, sigma2=sigma2, zi_logits=zi_logits).log_prob(torch.randn(B, G))
        assert lp.shape == (B, G)

    def test_finite_log_prob(self):
        mu = torch.zeros(B, G)
        sigma2 = torch.ones(B, G)
        zi_logits = torch.zeros(B, G)
        x = torch.randn(B, G)
        assert torch.isfinite(ZeroInflatedNormal(mu=mu, sigma2=sigma2, zi_logits=zi_logits).log_prob(x)).all()

    def test_high_zi_mean_reduced(self):
        mu = torch.tensor([2.0])
        sigma2 = torch.tensor([1.0])
        zi_logits = torch.tensor([10.0])  # high pi → mean ≈ 0
        d = ZeroInflatedNormal(mu=mu, sigma2=sigma2, zi_logits=zi_logits)
        assert d.mean.abs() < mu.abs()
