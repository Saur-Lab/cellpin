from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_COLORS = {
    "val_loss": "#2c3e50",
    "val_reconst_loss": "#c0392b",
    "val_kl_loss": "#2980b9",
    "val_kl_l_loss": "#8e44ad",
    "val_inv_loss": "#27ae60",
    "val_distill_loss": "#d35400",
    "val_snn_loss": "#16a085",
    "val_pearson_loss": "#f39c12",
}

_LABELS = {
    "val_loss": "Total val loss",
    "val_reconst_loss": "Reconstruction",
    "val_kl_loss": "KL (latent)",
    "val_kl_l_loss": "KL (library)",
    "val_inv_loss": "Invariance",
    "val_distill_loss": "Distillation",
    "val_snn_loss": "SNN alignment",
    "val_pearson_loss": "Pearson loss",
}

_DEFAULT_KEYS = ["val_loss", "val_reconst_loss", "val_inv_loss"]

_ALL_KEYS = [
    "val_loss",
    "val_reconst_loss",
    "val_kl_loss",
    "val_inv_loss",
    "val_distill_loss",
    "val_snn_loss",
    "val_kl_l_loss",
    "val_pearson_loss",
]

_SKIP = {"val_snn_temperature", "train_snn_temperature"}


def _find_csv(log_path: Path) -> Path:
    if log_path.is_file() and log_path.suffix == ".csv":
        return log_path
    candidates = sorted(log_path.rglob("metrics.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No metrics.csv found under {log_path}.\n"
            "Make sure you trained the model (CellPinTrainer writes a CSVLogger by default)."
        )
    return candidates[-1]


def _load_epoch_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "epoch" and c not in _SKIP]
    return df.groupby("epoch")[numeric_cols].mean(numeric_only=True).reset_index()


def losses(
    log_path: str | Path,
    keys: list[str] | None = None,
    *,
    smooth: int = 0,
    figsize: tuple[float, float] | None = None,
    save: str | Path | None = None,
) -> None:
    """Plot validation loss curves from a Lightning CSVLogger ``metrics.csv``.

    Parameters
    ----------
    log_path:
        Path to a ``metrics.csv`` file
    keys:
        Column names to plot, e.g. ``["val_loss", "val_reconst_loss"]``.
        Defaults to ``["val_loss", "val_reconst_loss", "val_inv_loss"]``.
        Pass ``"all"`` to show every available val loss.
    smooth:
        Width of a centered rolling-mean window. ``0`` (default) disables
        smoothing.
    figsize:
        ``(width, height)`` in inches. Auto-sized from the number of panels
        when omitted.
    save:
        If given, the figure is saved to this path at 300 dpi.
    """
    log_path = Path(log_path)
    csv_path = _find_csv(log_path)
    df = _load_epoch_df(csv_path)

    available_val = [c for c in df.columns if c.startswith("val_") and c not in _SKIP and not df[c].isna().all()]

    if keys is None:
        keys = [k for k in _DEFAULT_KEYS if k in available_val]
    elif keys == "all":
        keys = [k for k in _ALL_KEYS if k in available_val]
        keys += [k for k in available_val if k not in keys]
        keys = [k for k in keys if df[k].abs().max() > 1e-9]
    else:
        missing = [k for k in keys if k not in df.columns]
        if missing:
            raise ValueError(f"Keys not found in log: {missing}\nAvailable: {list(df.columns)}")

    if not keys:
        raise ValueError(f"No plottable loss columns found. Available: {list(df.columns)}")

    n = len(keys)
    w = figsize[0] if figsize else min(max(3.2 * n, 5.0), 16.0)
    h = figsize[1] if figsize else 3.2
    fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)
    axes_flat = axes.flatten()

    epochs = df["epoch"]

    def _smooth_s(s: pd.Series) -> pd.Series:
        if smooth <= 1:
            return s
        return s.rolling(window=smooth, center=True, min_periods=1).mean()

    for ax, key in zip(axes_flat, keys, strict=False):
        color = _COLORS.get(key, "#555555")
        label = _LABELS.get(key, key.replace("_", " "))
        series = df.get(key)

        if series is None or series.isna().all():
            ax.text(
                0.5, 0.5, "Not logged", ha="center", va="center", transform=ax.transAxes, color="#aaaaaa", fontsize=9
            )
        else:
            y = _smooth_s(series)
            ax.plot(epochs, y, color=color, linewidth=1.8)
            ax.set_xlim(epochs.iloc[0], epochs.iloc[-1])

        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300, bbox_inches="tight")

    plt.show()
