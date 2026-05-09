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

if TYPE_CHECKING:
    from cellpin.models import CellPin


def label_transfer(
    model: "CellPin",
    sc_adata: ad.AnnData,
    cell_type_col: str,
    sp_adata: ad.AnnData,
    conf_threshold: float = 0.0,
    k: int = 15,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[float, ad.AnnData]:
    from cellpin.dataset import scAnnDataset, stAnnDataset

    if cell_type_col not in sc_adata.obs.columns:
        raise ValueError(f"Column {cell_type_col!r} not found in sc_adata.obs")

    panel_genes = list(model.panel_gene_names)

    if "X_cellpin" not in sc_adata.obsm:
        print("[label_transfer] X_cellpin not found in sc_adata — embedding with default settings...")
        sc_ds = scAnnDataset(sc_adata, panel=panel_genes)
        sc_loader = DataLoader(sc_ds, batch_size=256, shuffle=False, num_workers=0)
        sc_imputed = model.impute(sc_loader)
        sc_adata.obsm["X_cellpin"] = sc_imputed.obsm["X_cellpin"]

    if "X_cellpin" not in sp_adata.obsm:
        print("[label_transfer] X_cellpin not found in sp_adata — embedding with default settings...")
        sp_ds = stAnnDataset(sp_adata, panel_genes=panel_genes)
        sp_loader = DataLoader(sp_ds, batch_size=256, shuffle=False, num_workers=0)
        sp_imputed = model.impute(sp_loader)
        sp_adata.obsm["X_cellpin"] = sp_imputed.obsm["X_cellpin"]

    X = np.array(sc_adata.obsm["X_cellpin"])
    y_raw = sc_adata.obs[cell_type_col].astype(str).values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print(f"\n[label_transfer] {n_classes} cell type classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    print(f"[label_transfer] Train: {len(X_train)} | Test: {len(X_test)}")

    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = float(accuracy_score(y_test, clf.predict(X_test)))
    print(f"[label_transfer] Test accuracy: {accuracy:.4f}")

    X_sp = np.array(sp_adata.obsm["X_cellpin"])
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
    return accuracy, sp_adata
