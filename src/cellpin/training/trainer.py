import logging
import warnings
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

# Suppress Lightning's INFO chatter (GPU available, LOCAL_RANK, litlogger tip, etc.)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Suppress known noisy warnings from Lightning / torch internals
warnings.filterwarnings("ignore", message=".*ipywidgets.*")
warnings.filterwarnings("ignore", message=".*num_workers.*bottleneck.*")
warnings.filterwarnings("ignore", message=".*`isinstance.*LeafSpec.*deprecated.*")
warnings.filterwarnings("ignore", message=".*set_float32_matmul_precision.*")


class CellPinTrainer:
    """Trainer class for CellPin models using PyTorch Lightning."""

    def __init__(
        self,
        max_epochs: int = 100,
        output_dir: str | Path = "./experiments/default",
        accelerator: str = "auto",
        devices: int | list = 1,
        precision: str = "32",
        accumulate_grad_batches: int = 1,
        gradient_clip_val: float | None = 1.0,
        enable_early_stopping: bool = True,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.0001,
        early_stopping_mode: str = "min",
        enable_checkpointing: bool = True,
        checkpoint_monitor: str = "val_loss",
        checkpoint_mode: str = "min",
        save_top_k: int = 3,
        log_every_n_steps: int = 10,
        enable_progress_bar: bool = True,
        deterministic: bool = False,
        seed: int | None = None,
        custom_callbacks: list | None = None,
        **trainer_kwargs,
    ):
        """Initializes the trainer.

        Args:
            max_epochs: Maximum number of epochs to train.
            output_dir: Directory to save outputs.
            accelerator: Accelerator to use (e.g., 'cpu', 'gpu').
            devices: Number of devices or list of device IDs.
            precision: Precision for training ('16', '32', etc.).
            accumulate_grad_batches: Number of batches to accumulate gradients.
            gradient_clip_val: Value to clip gradients.
            enable_early_stopping: Whether to enable early stopping.
            early_stopping_patience: Patience for early stopping.
            early_stopping_min_delta: Minimum delta for early stopping.
            early_stopping_mode: Mode for early stopping ('min' or 'max').
            enable_checkpointing: Whether to enable checkpointing.
            checkpoint_monitor: Metric to monitor for checkpointing.
            checkpoint_mode: Mode for checkpointing ('min' or 'max').
            save_top_k: Number of top models to save.
            log_every_n_steps: Log every n steps.
            enable_progress_bar: Whether to enable progress bar.
            deterministic: Whether to make training deterministic.
            seed: Random seed for reproducibility.
            custom_callbacks: List of custom callbacks.
            **trainer_kwargs: Additional arguments for PyTorch Lightning Trainer.
        """
        self.max_epochs = max_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seed for reproducibility
        if seed is not None:
            pl.seed_everything(seed, workers=True)

        # Initialize callbacks
        self.callbacks = []

        # Early stopping callback
        if enable_early_stopping:
            early_stop_callback = EarlyStopping(
                monitor=checkpoint_monitor,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                mode=early_stopping_mode,
                verbose=False,
            )
            self.callbacks.append(early_stop_callback)

        # Checkpointing callbacks
        if enable_checkpointing:
            # Best model checkpoint
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.output_dir / "checkpoints",
                filename="best-{epoch:02d}-{val_loss:.4f}",
                monitor=checkpoint_monitor,
                mode=checkpoint_mode,
                save_top_k=save_top_k,
                save_last=True,
                verbose=False,
            )
            self.callbacks.append(checkpoint_callback)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        self.callbacks.append(lr_monitor)

        # Progress bar — TQDMProgressBar works in both terminal and Jupyter
        if enable_progress_bar:
            progress_bar = TQDMProgressBar(leave=False)
            self.callbacks.append(progress_bar)

        # Add any custom callbacks
        if custom_callbacks:
            self.callbacks.extend(custom_callbacks)

        # Logger
        self.logger = TensorBoardLogger(
            save_dir=str(self.output_dir),
            name="logs",
            default_hp_metric=False,
        )

        # Initialize trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=precision,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_val=gradient_clip_val,
            callbacks=self.callbacks,
            logger=self.logger,
            log_every_n_steps=log_every_n_steps,
            deterministic=deterministic,
            enable_progress_bar=enable_progress_bar,
            enable_checkpointing=enable_checkpointing,
            enable_model_summary=False,
            **trainer_kwargs,
        )

    def fit(
        self,
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        ckpt_path: str | None = None,
    ) -> None:
        """Train the model.

        Args:
            model: LightningModule to train.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data (optional).
            ckpt_path: Path to checkpoint to resume training from.
        """
        self.trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=ckpt_path,
        )

    def validate(
        self,
        model: pl.LightningModule,
        val_dataloader: DataLoader,
        ckpt_path: str | None = None,
    ) -> list[dict[str, float]]:
        """Run validation.

        Args:
            model: LightningModule to validate.
            val_dataloader: DataLoader for validation data.
            ckpt_path: Path to checkpoint to load.

        Returns
        -------
            list[dict[str, float]]: List of validation metrics dictionaries.
        """
        return self.trainer.validate(
            model,
            dataloaders=val_dataloader,
            ckpt_path=ckpt_path,
        )

    def test(
        self,
        model: pl.LightningModule,
        test_dataloader: DataLoader,
        ckpt_path: str | None = None,
    ) -> list[dict[str, float]]:
        """
        Run testing on the model.

        Args:
            model: LightningModule to test
            test_dataloader: DataLoader for test data
            ckpt_path: Path to checkpoint to load

        Returns
        -------
            List of test metrics dictionaries
        """
        return self.trainer.test(
            model,
            dataloaders=test_dataloader,
            ckpt_path=ckpt_path,
        )

    @property
    def best_model_path(self) -> str | None:
        """Get path to the best saved checkpoint."""
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback.best_model_path
        return None

    @property
    def last_model_path(self) -> str | None:
        """Get path to the last saved checkpoint."""
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                return callback.last_model_path
        return None


