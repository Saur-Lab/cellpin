from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from cellpin._sdata_utils import _resolve_sdata

if TYPE_CHECKING:
    from cellpin.models import CellPin


def label_transfer(
    model: CellPin,
    sc_adata: ad.AnnData,
    cell_type_col: str,
    sp_adata: ad.AnnData,
    conf_threshold: float = 0.0,
    k: int = 15,
    test_size: float = 0.2,
    random_state: int = 42,
    table_key: str = "table",
    emb_key: str | None = None,
) -> tuple[float, ad.AnnData]:
    """Transfer cell type labels from scRNA to spatial data via kNN in embedding space.

    Works with both the standard CellPin VAE path (``model.fit()``) and the
    atlas-matching path (``model.match_emb()``).

    If ``emb_key`` is not provided it is auto-detected: ``"X_cellpin_match"`` when
    ``model.atlas_net`` is set (atlas-matching path), ``"X_cellpin"`` otherwise.

    When the chosen embedding is absent from ``sc_adata.obsm`` or
    ``sp_adata.obsm`` it is computed automatically:

    * VAE path  → :meth:`~cellpin.models.CellPin.impute`
    * Atlas path → :meth:`~cellpin.models.CellPin.embed_atlas`

    Args:
        model: Trained :class:`~cellpin.models.CellPin` instance.
        sc_adata: Single-cell AnnData with ground-truth cell type labels.
        cell_type_col: Column in ``sc_adata.obs`` containing the cell type labels.
        sp_adata: Spatial AnnData (or SpatialData) to annotate. Modified in-place:
            ``sp_adata.obs["cellpin_annotation"]`` and
            ``sp_adata.obs["cellpin_annotation_certainty"]`` are written.
        conf_threshold: Min max-class probability to assign a label (default 0.0 =
            annotate all). Cells below this threshold receive the label ``"Unknown"``.
        k: Number of nearest neighbours (default 15).
        test_size: Fraction of scRNA cells held out for evaluation (default 0.2).
        random_state: Random seed for the train/test split (default 42).
        table_key: Table key when ``sp_adata`` is a ``SpatialData`` object.
        emb_key: Embedding key to use in ``obsm`` for both ``sc_adata`` and
            ``sp_adata``. Defaults to ``"X_cellpin_match"`` on the atlas path
            and ``"X_cellpin"`` on the VAE path.

    Returns:
        Tuple of ``(test_accuracy, sp_adata)``.
    """
    from cellpin.dataset import scAnnDataset, stAnnDataset

    sp_adata, sdata = _resolve_sdata(sp_adata, table_key)

    if cell_type_col not in sc_adata.obs.columns:
        raise ValueError(f"Column {cell_type_col!r} not found in sc_adata.obs")

    # Auto-detect embedding path.
    use_atlas = model.atlas_net is not None
    if emb_key is None:
        emb_key = "X_cellpin_match" if use_atlas else "X_cellpin"
    print(f"[label_transfer] Using embedding: {emb_key}")

    panel_genes = list(model.panel_gene_names)

    if emb_key not in sc_adata.obsm:
        if use_atlas:
            print(f"[label_transfer] {emb_key} not found in sc_adata — embedding via atlas network...")
            sc_ds = scAnnDataset(sc_adata, panel=panel_genes)
            sc_loader = DataLoader(sc_ds, batch_size=256, shuffle=False, num_workers=0)
            sc_adata.obsm[emb_key] = model.embed_atlas(sc_loader)
        else:
            print(f"[label_transfer] {emb_key} not found in sc_adata — embedding with default settings...")
            sc_ds = scAnnDataset(sc_adata, panel=panel_genes)
            sc_loader = DataLoader(sc_ds, batch_size=256, shuffle=False, num_workers=0)
            sc_imputed = model.impute(sc_loader)
            sc_adata.obsm[emb_key] = sc_imputed.obsm["X_cellpin"]

    if emb_key not in sp_adata.obsm:
        if use_atlas:
            print(f"[label_transfer] {emb_key} not found in sp_adata — embedding via atlas network...")
            sp_ds = stAnnDataset(sp_adata, panel_genes=panel_genes)
            sp_loader = DataLoader(sp_ds, batch_size=256, shuffle=False, num_workers=0)
            sp_adata.obsm[emb_key] = model.embed_atlas(sp_loader)
        else:
            print(f"[label_transfer] {emb_key} not found in sp_adata — embedding with default settings...")
            sp_ds = stAnnDataset(sp_adata, panel_genes=panel_genes)
            sp_loader = DataLoader(sp_ds, batch_size=256, shuffle=False, num_workers=0)
            sp_imputed = model.impute(sp_loader)
            sp_adata.obsm[emb_key] = sp_imputed.obsm["X_cellpin"]

    X = np.array(sc_adata.obsm[emb_key])
    y_raw = sc_adata.obs[cell_type_col].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"\n[label_transfer] {n_classes} cell type classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[label_transfer] Train: {len(X_train)} | Test: {len(X_test)}")

    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))
    print(f"[label_transfer] Test accuracy: {accuracy:.4f}")

    X_sp = np.array(sp_adata.obsm[emb_key])
    proba = clf.predict_proba(X_sp)
    max_proba = proba.max(axis=1)
    pred_labels = le.inverse_transform(proba.argmax(axis=1)).astype(object)

    if conf_threshold > 0.0:
        uncertain = max_proba < conf_threshold
        n_uncertain = uncertain.sum()
        pred_labels[uncertain] = "Unknown"
        print(
            f"[label_transfer] {n_uncertain} / {len(pred_labels)} spatial cells "
            f"labelled 'Unknown' (certainty < {conf_threshold})"
        )

    sp_adata.obs["cellpin_annotation"] = pd.Categorical(pred_labels)
    sp_adata.obs["cellpin_annotation_certainty"] = max_proba.astype(np.float32)

    print("[label_transfer] Annotation complete. Annotations stored in sp_adata.obs['cellpin_annotation']")
    if sdata is not None:
        return accuracy, sdata
    return accuracy, sp_adata
