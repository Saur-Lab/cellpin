import torch
from torch.utils.data import TensorDataset

from cellpin.models.utils.model_helpers import build_data_loaders, load_config_and_checkpoint, save_checkpoint


def test_load_config_defaults():
    params, state_dict = load_config_and_checkpoint()
    assert "lr" in params
    assert "weight_decay" in params
    assert state_dict is None


def test_load_config_dict():
    config = {"n_latent": 64}
    params, state_dict = load_config_and_checkpoint(config=config)
    assert params["n_latent"] == 64
    assert state_dict is None


def test_save_and_load_checkpoint(tmp_path):
    ckpt_path = tmp_path / "model.ckpt"
    state_dict = {"layer.weight": torch.tensor([1.0])}
    hparams = {"n_latent": 32}

    save_checkpoint(ckpt_path, state_dict, hparams)
    assert ckpt_path.exists()

    params, loaded_state_dict = load_config_and_checkpoint(checkpoint=ckpt_path)
    assert params["n_latent"] == 32
    assert "layer.weight" in loaded_state_dict
    assert torch.equal(loaded_state_dict["layer.weight"], state_dict["layer.weight"])


def test_build_data_loaders():
    X = torch.randn(10, 5)
    dataset = TensorDataset(X)
    train_loader, val_loader = build_data_loaders(dataset, train_size=0.5, batch_size=2)

    assert len(train_loader.dataset) == 5
    assert len(val_loader.dataset) == 5
    assert train_loader.batch_size == 2
