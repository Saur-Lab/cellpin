import pytest
import torch

from cellpin.models.vae import CellPinVAE

N_FULL = 20
N_PANEL = 8
B = 4


@pytest.fixture
def vae():
    return CellPinVAE(
        n_input_full=N_FULL,
        n_input_panel=N_PANEL,
        n_hidden=32,
        n_latent=8,
        n_layers_encoder=2,
        n_layers_decoder=1,
        reconstruction_loss="nb",
    )


@pytest.fixture
def batch_tensors():
    torch.manual_seed(0)
    x_full = torch.rand(B, N_FULL) * 5
    x_panel = x_full[:, :N_PANEL]
    return x_full, x_panel


class TestInference:
    def test_panel_view_output_shapes(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        out = vae.inference(x_full, x_panel=x_panel, encoder_view="panel")
        assert out["z"].shape == (B, 8)
        assert out["px_rate"].shape == (B, N_FULL)
        assert out["qz_m"].shape == (B, 8)
        assert out["qz_v"].shape == (B, 8)
        assert out["library"].shape == (B, 1)

    def test_full_view_output_shapes(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        out = vae.inference(x_full, x_panel=x_panel, encoder_view="full")
        assert out["z"].shape == (B, 8)
        assert out["px_rate"].shape == (B, N_FULL)

    def test_px_rate_non_negative(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        out = vae.inference(x_full, x_panel=x_panel)
        assert (out["px_rate"] >= 0).all()

    def test_all_outputs_finite(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        out = vae.inference(x_full, x_panel=x_panel)
        for key in ("z", "px_rate", "qz_m", "qz_v", "library"):
            assert torch.isfinite(out[key]).all(), f"{key} has non-finite values"

    def test_qz_v_positive(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        out = vae.inference(x_full, x_panel=x_panel)
        assert (out["qz_v"] > 0).all()

    def test_invalid_encoder_view_raises(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        with pytest.raises(ValueError, match="Unknown encoder_view"):
            vae.inference(x_full, x_panel=x_panel, encoder_view="invalid")


class TestForward:
    def test_returns_three_elements(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        local_l_mean = torch.zeros(B, 1)
        local_l_var = torch.ones(B, 1)
        result = vae.forward(x_full, local_l_mean, local_l_var, x_panel=x_panel, encoder_view="full")
        assert len(result) == 3

    def test_reconst_kl_l_shape(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        local_l_mean = torch.zeros(B, 1)
        local_l_var = torch.ones(B, 1)
        reconst_kl_l, kl_z, _ = vae.forward(x_full, local_l_mean, local_l_var, x_panel=x_panel, encoder_view="full")
        assert reconst_kl_l.shape == (B,)
        assert kl_z.shape == (B,)

    def test_reconst_kl_l_finite(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        local_l_mean = torch.zeros(B, 1)
        local_l_var = torch.ones(B, 1)
        reconst_kl_l, kl_z, _ = vae.forward(x_full, local_l_mean, local_l_var, x_panel=x_panel, encoder_view="full")
        assert torch.isfinite(reconst_kl_l).all()
        assert torch.isfinite(kl_z).all()

    def test_kl_z_non_negative(self, vae, batch_tensors):
        x_full, x_panel = batch_tensors
        local_l_mean = torch.zeros(B, 1)
        local_l_var = torch.ones(B, 1)
        _, kl_z, _ = vae.forward(x_full, local_l_mean, local_l_var, x_panel=x_panel, encoder_view="full")
        assert (kl_z >= 0).all()


class TestReconstructionLoss:
    @pytest.mark.parametrize("loss_type", ["nb", "zinb", "poisson"])
    def test_shape_and_finite(self, loss_type):
        vae = CellPinVAE(
            n_input_full=N_FULL,
            n_input_panel=N_PANEL,
            n_hidden=32,
            n_latent=8,
            n_layers_encoder=2,
            n_layers_decoder=1,
            reconstruction_loss=loss_type,
        )
        x = torch.rand(B, N_FULL) * 5
        px_rate = torch.rand(B, N_FULL).abs() + 0.1
        px_r = torch.exp(vae.px_r)
        px_dropout = torch.zeros(B, N_FULL)
        loss = vae.get_reconstruction_loss(x, px_rate, px_r, px_dropout)
        assert loss.shape == (B,)
        assert torch.isfinite(loss).all()

    def test_unknown_loss_raises(self):
        vae = CellPinVAE(
            n_input_full=N_FULL,
            n_input_panel=N_PANEL,
            n_hidden=32,
            n_latent=8,
            n_layers_encoder=2,
            n_layers_decoder=1,
        )
        vae.reconstruction_loss = "bad"
        x = torch.rand(B, N_FULL)
        px_rate = torch.rand(B, N_FULL).abs() + 0.1
        with pytest.raises(ValueError, match="Unknown reconstruction_loss"):
            vae.get_reconstruction_loss(x, px_rate, torch.ones(N_FULL), torch.zeros(B, N_FULL))


class TestExcludePanel:
    def test_inference_still_runs(self):
        panel_idx = list(range(N_PANEL))
        vae = CellPinVAE(
            n_input_full=N_FULL,
            n_input_panel=N_PANEL,
            panel_idx=panel_idx,
            exclude_panel=True,
            n_hidden=32,
            n_latent=8,
            n_layers_encoder=2,
            n_layers_decoder=1,
            reconstruction_loss="nb",
        )
        x_full = torch.rand(B, N_FULL) * 5
        x_panel = x_full[:, :N_PANEL]
        out = vae.inference(x_full, x_panel=x_panel, encoder_view="full")
        assert out["z"].shape == (B, 8)
        assert torch.isfinite(out["z"]).all()
