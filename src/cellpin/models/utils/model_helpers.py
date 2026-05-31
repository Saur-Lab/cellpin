from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, random_split

DEFAULT_PARAMS = {
    # Architecture
    "n_latent": 192,
    "n_hidden": 1024,
    "encoder_layers": 16,
    "encoder_dropout": 0.2,
    "drop_path_rate": 0.15,
    "ffn_expansion": 2,
    "layer_scale_init": 0.0014,
    "decoder_layers": 2,
    "reconstruction_loss": "nb",
    "log_variational": True,
    # Augmentation
    "use_panel_only": True,
    "encoder_noise_std": 0.1,
    "panel_mixup_alpha": 0.1,
    "poisson_resample_rate": 0.1,
    # Loss weights
    "lambda_recon": 1.17,
    "kl_weight": 0.08,
    "kl_warmup_epochs": 20,
    "lambda_inv": 20.0,
    "lambda_snn": 0.085,
    "lambda_distill": 1.0,
    "lambda_pearson": 0.05,
    "reconstruct_panel": True,
    "distillation_mode": "mse",
    "snn_temperature_init": 0.173,
    "exclude_panel": False,
    # Optimiser
    "lr": 0.00021,
    "weight_decay": 1.3e-6,
}


def load_config_and_checkpoint(config: dict[str, Any] | None = None, checkpoint: Path | None = None):
    """Load configuration parameters and model state dict from config or checkpoint.

    Args:
        config: Configuration dictionary or path to YAML file. If None, uses defaults.
        checkpoint: Path to checkpoint file. If provided, loads params from it.

    Returns:
    -------
        Tuple of (params dict, loaded_state_dict or None).
    """
    params = dict(DEFAULT_PARAMS)

    loaded_state_dict = None

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        if "hyperparameters" in ckpt:
            params.update(ckpt["hyperparameters"])
        loaded_state_dict = ckpt.get("state_dict")
    elif config is not None:
        if isinstance(config, str | Path):
            with open(config) as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = config
        params.update(config_dict)

    return params, loaded_state_dict


def save_checkpoint(path: Path, state_dict: dict[str, Any], hyperparameters: dict[str, Any]):
    """Save model checkpoint with state dict and hyperparameters.

    Args:
        path: Path to save the checkpoint.
        state_dict: Model state dictionary.
        hyperparameters: Hyperparameters dictionary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "hyperparameters": dict(hyperparameters)}, path)


def build_data_loaders(
    dataset, train_size: float = 0.8, batch_size: int = 128, num_workers: int = 4, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """Build train and validation data loaders from dataset.

    Args:
        dataset: The dataset to split.
        train_size: Fraction of data for training.
        batch_size: Batch size for loaders.
        num_workers: Number of workers for data loading.
        seed: Random seed for splitting.

    Returns:
    -------
        Tuple of (train_loader, val_loader).
    """
    n_total = len(dataset)
    n_train = int(round(train_size * n_total))
    n_val = n_total - n_train
    if n_val <= 0:
        raise ValueError("Train size must be less than 1.0")
    train_dataset, val_dataset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader
