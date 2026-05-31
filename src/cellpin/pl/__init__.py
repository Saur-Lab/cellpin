from __future__ import annotations

from typing import TYPE_CHECKING

from cellpin.pl._losses import losses

if TYPE_CHECKING:
    from cellpin.models import CellPin


class PlotAccessor:
    """Plotting accessor attached to a :class:`~cellpin.models.CellPin` instance.

    Access via ``model.pl``.
    """

    def __init__(self, model: CellPin):
        self._model = model

    def losses(
        self,
        log_path: str | None = None,
        keys: list[str] | None = None,
        *,
        smooth: int = 0,
        figsize: tuple[float, float] | None = None,
        save: str | None = None,
    ) -> None:
        """Plot validation loss curves for this model.

        Args:
            log_path: Override the log directory to read from. When omitted the
                trainer output directory stored on the model is used (set
                automatically after calling
                :meth:`~cellpin.models.CellPin.fit`,
                :meth:`~cellpin.models.CellPin.train_model`, or
                :meth:`~cellpin.models.CellPin.pretrain_model`).
            keys: Which loss columns to plot, e.g.
                ``["val_loss", "val_reconst_loss"]``. Defaults to
                ``["val_loss", "val_reconst_loss", "val_inv_loss"]``. Pass
                ``"all"`` to show every available val loss.
            smooth: Centered rolling-mean window width (0 = off).
            figsize: ``(width, height)`` in inches. Auto-sized when omitted.
            save: Save the figure to this path at 300 dpi when given.
        """
        if log_path is None:
            log_path = getattr(self._model, "_train_output_dir", None) or getattr(
                self._model, "_pretrain_output_dir", None
            )
            if log_path is None:
                raise RuntimeError(
                    "No log directory found on the model.\n"
                    "Either train the model first (fit / train_model / pretrain_model) "
                    "or pass log_path explicitly."
                )
        return losses(log_path, keys=keys, smooth=smooth, figsize=figsize, save=save)


__all__ = ["losses", "PlotAccessor"]
