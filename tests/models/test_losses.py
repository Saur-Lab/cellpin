from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from cellpin.models.cellpin_model import CellPin, soft_nn_loss


def _make_model(sc_dataset, **config_overrides):
    config = {"n_latent": 8, "n_hidden": 32, "encoder_layers": 2, "decoder_layers": 1}
    config.update(config_overrides)
    return CellPin(sc_dataset=sc_dataset, config=config)


class TestSoftNNLoss:
    def test_output_is_scalar(self):
        z_panel = torch.randn(4, 8)
        z_full = torch.randn(4, 8)
        assert soft_nn_loss(z_panel, z_full).shape == ()

    def test_finite_and_non_negative(self):
        z_panel = torch.randn(6, 16)
        z_full = torch.randn(6, 16)
        loss = soft_nn_loss(z_panel, z_full)
        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_matched_embeddings_less_than_random(self):
        torch.manual_seed(0)
        z = torch.randn(8, 64)
        z_random = torch.randn(8, 64)
        loss_matched = soft_nn_loss(z, z)
        loss_random = soft_nn_loss(z, z_random)
        assert loss_matched < loss_random

    def test_symmetric(self):
        torch.manual_seed(1)
        z1 = torch.randn(6, 12)
        z2 = torch.randn(6, 12)
        assert torch.allclose(soft_nn_loss(z1, z2), soft_nn_loss(z2, z1), atol=1e-6)

    def test_temperature_scales_sharpness(self):
        torch.manual_seed(2)
        z = torch.randn(8, 16)
        loss_sharp = soft_nn_loss(z, z, temperature=0.01)
        loss_flat = soft_nn_loss(z, z, temperature=1.0)
        # Sharper temperature → closer to 0 for matched pairs
        assert loss_sharp <= loss_flat


class TestPearsonLoss:
    def test_output_scalar(self):
        x = torch.randn(8, 4)
        assert CellPin._pearson_loss(x, torch.randn(8, 4)).shape == ()

    def test_perfect_correlation_zero(self):
        x = torch.randn(16, 10)
        assert CellPin._pearson_loss(x, x).item() < 1e-5

    def test_anticorrelation_gives_two(self):
        x = torch.randn(32, 10)
        loss = CellPin._pearson_loss(x, -x)
        assert abs(loss.item() - 2.0) < 0.01

    def test_range(self):
        torch.manual_seed(0)
        x = torch.randn(64, 20)
        y = torch.randn(64, 20)
        loss = CellPin._pearson_loss(x, y)
        assert 0.0 <= loss.item() <= 2.0


class TestKLAnnealing:
    def test_no_warmup_returns_one(self, sc_dataset):
        model = _make_model(sc_dataset, kl_warmup_epochs=0)
        assert model._kl_annealing_weight() == 1.0

    def test_negative_warmup_returns_one(self, sc_dataset):
        model = _make_model(sc_dataset, kl_warmup_epochs=-1)
        assert model._kl_annealing_weight() == 1.0

    def test_at_epoch_zero_returns_zero(self, sc_dataset):
        model = _make_model(sc_dataset, kl_warmup_epochs=10)
        # current_epoch = 0 before training starts
        assert model._kl_annealing_weight() == 0.0

    def test_clamped_at_one_after_warmup(self, sc_dataset):
        model = _make_model(sc_dataset, kl_warmup_epochs=5)
        with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=100):
            assert model._kl_annealing_weight() == 1.0

    def test_midpoint(self, sc_dataset):
        model = _make_model(sc_dataset, kl_warmup_epochs=10)
        with patch.object(type(model), "current_epoch", new_callable=PropertyMock, return_value=5):
            assert model._kl_annealing_weight() == pytest.approx(0.5)


class TestMixupPanel:
    def test_disabled_returns_unchanged(self, sc_dataset):
        model = _make_model(sc_dataset, panel_mixup_alpha=0.0)
        model.train()
        x = torch.rand(4, 8)
        assert torch.allclose(model._mixup_panel(x), x)

    def test_eval_mode_returns_unchanged(self, sc_dataset):
        model = _make_model(sc_dataset, panel_mixup_alpha=0.5)
        model.eval()
        x = torch.rand(4, 8)
        assert torch.allclose(model._mixup_panel(x), x)

    def test_train_mode_blends_values(self, sc_dataset):
        model = _make_model(sc_dataset, panel_mixup_alpha=0.5)
        model.train()
        torch.manual_seed(0)
        x = torch.rand(16, 8)
        out = model._mixup_panel(x)
        assert not torch.allclose(out, x)

    def test_output_non_negative(self, sc_dataset):
        model = _make_model(sc_dataset, panel_mixup_alpha=0.3)
        model.train()
        x = torch.rand(8, 8).abs()
        out = model._mixup_panel(x)
        assert (out >= 0).all()


class TestPoissonResample:
    def test_disabled_returns_unchanged(self, sc_dataset):
        model = _make_model(sc_dataset, poisson_resample_rate=0.0)
        model.train()
        x = torch.rand(4, 8) * 10
        assert torch.allclose(model._poisson_resample_panel(x), x)

    def test_eval_mode_returns_unchanged(self, sc_dataset):
        model = _make_model(sc_dataset, poisson_resample_rate=0.5)
        model.eval()
        x = torch.rand(4, 8) * 10
        assert torch.allclose(model._poisson_resample_panel(x), x)

    def test_output_non_negative(self, sc_dataset):
        model = _make_model(sc_dataset, poisson_resample_rate=0.4)
        model.train()
        x = torch.rand(8, 8) * 20
        out = model._poisson_resample_panel(x)
        assert (out >= 0).all()

    def test_output_integer_valued(self, sc_dataset):
        model = _make_model(sc_dataset, poisson_resample_rate=0.4)
        model.train()
        x = torch.rand(8, 8) * 20
        out = model._poisson_resample_panel(x)
        assert torch.allclose(out, out.floor())


class TestKLFreeBits:
    def test_disabled_by_default(self, sc_dataset):
        model = _make_model(sc_dataset)
        assert model.kl_free_bits == 0.0

    def test_disabled_matches_plain_kl(self, sc_dataset):
        model = _make_model(sc_dataset, kl_warmup_epochs=0)
        torch.manual_seed(0)
        qz_m = torch.randn(16, 4) * 0.01
        qz_v = torch.full((16, 4), 0.999)
        kl_per_dim = kl(Normal(qz_m, qz_v.sqrt()), Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v)))
        expected = kl_per_dim.sum(dim=1).mean()
        assert model._kl_z_loss(qz_m, qz_v).item() == pytest.approx(expected.item(), rel=1e-5)

    def test_floor_raises_near_collapsed_posterior(self, sc_dataset):
        model = _make_model(sc_dataset, kl_warmup_epochs=0, kl_free_bits=0.5)
        qz_m = torch.randn(64, 10) * 0.01
        qz_v = torch.full((64, 10), 0.999)
        loss = model._kl_z_loss(qz_m, qz_v)
        assert loss.item() >= 10 * 0.5 - 1e-4

    def test_floor_is_noop_once_kl_already_above_it(self, sc_dataset):
        torch.manual_seed(0)
        qz_m = torch.randn(64, 10) * 3.0
        qz_v = torch.full((64, 10), 0.5)
        model = _make_model(sc_dataset, kl_warmup_epochs=0, kl_free_bits=0.01)
        floored = model._kl_z_loss(qz_m, qz_v)
        model.kl_free_bits = 0.0
        unfloored = model._kl_z_loss(qz_m, qz_v)
        assert floored.item() == pytest.approx(unfloored.item(), rel=1e-4)
