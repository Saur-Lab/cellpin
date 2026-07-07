from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad

from cellpin.tl._label_transfer import label_transfer as _label_transfer_fn

if TYPE_CHECKING:
    from cellpin.models import CellPin


class TLAccessor:
    """Tools accessor attached to a :class:`~cellpin.models.CellPin` instance.

    Access via ``model.tl``.
    """

    def __init__(self, model: CellPin):
        self._model = model

    def label_transfer(
        self,
        sc_adata: ad.AnnData,
        cell_type_col: str,
        sp_adata: ad.AnnData,
        conf_threshold: float = 0.0,
        k: int = 15,
        test_size: float = 0.2,
        random_state: int = 42,
        emb_key: str | None = None,
    ) -> tuple[float, ad.AnnData]:
        """Transfer cell type labels from scRNA to spatial data via kNN in embedding space.

        Works with both the standard VAE path (``model.fit()``) and the atlas-matching
        path (``model.match_emb()``).  ``emb_key`` is auto-detected when omitted:
        ``"X_cellpin_match"`` after ``match_emb()``, ``"X_cellpin"`` otherwise.

        If the chosen embedding is absent from either adata it is computed automatically.

        Args:
            sc_adata: Single-cell AnnData with ground-truth cell type labels.
            cell_type_col: Column in ``sc_adata.obs`` containing the cell type labels.
            sp_adata: Spatial AnnData to annotate. Modified in-place:
                ``sp_adata.obs["cellpin_annotation"]`` and
                ``sp_adata.obs["cellpin_annotation_certainty"]`` are written.
            conf_threshold: Min max-class probability to assign a label (default 0.0 =
                annotate all). Cells below this threshold receive the label ``"Unknown"``.
            k: Number of nearest neighbours (default 15).
            test_size: Fraction of scRNA cells held out for evaluation (default 0.2).
            random_state: Random seed for the train/test split (default 42).
            emb_key: Embedding key to use (auto-detected when ``None``).

        Returns:
            Tuple of ``(test_accuracy, sp_adata)``.
        """
        return _label_transfer_fn(
            self._model,
            sc_adata,
            cell_type_col,
            sp_adata,
            conf_threshold=conf_threshold,
            k=k,
            test_size=test_size,
            random_state=random_state,
            emb_key=emb_key,
        )


label_transfer = _label_transfer_fn

__all__ = ["TLAccessor", "label_transfer"]
