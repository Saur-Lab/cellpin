"""Tests for the trainer wrapper using the current CellPin API."""

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from cellpin.training import CellPinTrainer


def test_trainer_initialization(tmp_path):
    trainer = CellPinTrainer(max_epochs=10, output_dir=str(tmp_path / "run"))
    assert trainer.max_epochs == 10
    assert trainer.trainer is not None
    assert len(trainer.callbacks) > 0


def test_trainer_early_stopping(tmp_path):
    trainer = CellPinTrainer(
        max_epochs=10,
        output_dir=str(tmp_path / "run"),
        enable_early_stopping=True,
        early_stopping_patience=3,
    )
    assert any(isinstance(cb, EarlyStopping) for cb in trainer.callbacks)


def test_trainer_no_checkpointing(tmp_path):
    trainer = CellPinTrainer(
        max_epochs=10,
        output_dir=str(tmp_path / "run"),
        enable_checkpointing=False,
    )
    assert not any(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks)
