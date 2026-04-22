#!/usr/bin/env python3
"""
run_cellpin_holdout.py
======================
Hold-out gene evaluation utilities for CellPin.

Reserves a subset of spatial genes before training, then evaluates
imputation quality on those held-out genes after training completes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

import cellpin.pp
from cellpin.models import CellPin


@dataclass
class PreparedData:
    adata_sc: ad.AnnData
    adata_st: ad.AnnData   # spatial with test_genes removed — training panel
    adata_sp: ad.AnnData   # full spatial — hold-out evaluation ground truth
    test_genes: list[str] = field(default_factory=list)


def prepare_holdout_data(
    adata_path: str,
    spatial_path: str,
    n_cells: int = 50_000,
    n_genes: int = 2000,
    n_holdout: int = 50,
    required_holdout: list[str] | None = None,
    seed: int = 42,
    layer: str | None = None,
) -> PreparedData:
    """Load data, subsample, and hold out ``n_holdout`` spatial genes for evaluation."""
    rng = np.random.default_rng(seed)

    adata_sc = ad.read_h5ad(adata_path)
    adata_sp_full = ad.read_h5ad(spatial_path)

    if adata_sc.n_obs > n_cells:
        idx = rng.choice(adata_sc.n_obs, n_cells, replace=False)
        adata_sc = adata_sc[idx].copy()

    if n_genes and adata_sc.n_vars > n_genes:
        try:
            import scanpy as sc
            sc.pp.highly_variable_genes(
                adata_sc, n_top_genes=n_genes, flavor="seurat_v3", layer=layer,
            )
            adata_sc = adata_sc[:, adata_sc.var["highly_variable"]].copy()
        except Exception as exc:
            warnings.warn(
                f"HVG selection failed ({exc}); using all {adata_sc.n_vars} genes.",
                stacklevel=2,
            )

    sc_genes = set(adata_sc.var_names.tolist())
    sp_genes_overlap = [g for g in adata_sp_full.var_names.tolist() if g in sc_genes]

    if len(sp_genes_overlap) < n_holdout + 5:
        raise ValueError(
            f"Only {len(sp_genes_overlap)} overlapping sc/spatial genes — "
            f"insufficient for {n_holdout} hold-out genes."
        )

    required = [g for g in (required_holdout or []) if g in set(sp_genes_overlap)]
    remaining = [g for g in sp_genes_overlap if g not in set(required)]
    n_extra = max(0, n_holdout - len(required))
    extra = rng.choice(remaining, min(n_extra, len(remaining)), replace=False).tolist()
    test_genes = required + extra

    train_sp_genes = [g for g in sp_genes_overlap if g not in set(test_genes)]
    if len(train_sp_genes) < 5:
        raise ValueError("Too few training panel genes after hold-out split.")

    adata_st = adata_sp_full[:, train_sp_genes].copy()
    adata_sp = adata_sp_full.copy()

    return PreparedData(
        adata_sc=adata_sc,
        adata_st=adata_st,
        adata_sp=adata_sp,
        test_genes=test_genes,
    )


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    gene_names: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Per-gene Pearson, RMSE, JS divergence, SSIM; returns (summary_dict, per_gene_df)."""
    n_genes = y_pred.shape[1]
    pearsons, rmses, jss, ssims = [], [], [], []

    for g in range(n_genes):
        pred = y_pred[:, g].astype(float)
        true = y_true[:, g].astype(float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pred.std() < 1e-12 or true.std() < 1e-12:
                r = 0.0
            else:
                r, _ = pearsonr(pred, true)
                r = float(r) if np.isfinite(r) else 0.0
        pearsons.append(r)

        rmses.append(float(np.sqrt(np.mean((pred - true) ** 2))))

        p = pred - pred.min() + 1e-8
        q = true - true.min() + 1e-8
        jss.append(float(jensenshannon(p / p.sum(), q / q.sum())))

        mu1, mu2 = pred.mean(), true.mean()
        s1 = pred.std() + 1e-8
        s2 = true.std() + 1e-8
        s12 = float(np.cov(pred, true)[0, 1])
        dyn = max(abs(mu1), abs(mu2), 1.0)
        c1, c2 = (0.01 * dyn) ** 2, (0.03 * max(s1, s2)) ** 2
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / (
            (mu1 ** 2 + mu2 ** 2 + c1) * (s1 ** 2 + s2 ** 2 + c2)
        )
        ssims.append(float(ssim_val) if np.isfinite(ssim_val) else 0.0)

    per_gene = pd.DataFrame({
        "gene":    gene_names,
        "Pearson": pearsons,
        "RMSE":    rmses,
        "JS":      jss,
        "SSIM":    ssims,
    })
    summary = {
        "Pearson_mean": float(np.mean(pearsons)),
        "RMSE_mean":    float(np.mean(rmses)),
        "JS_mean":      float(np.mean(jss)),
        "SSIM_mean":    float(np.mean(ssims)),
    }
    return summary, per_gene


def run_inference_and_evaluate(
    model: CellPin,
    prepared: PreparedData,
    batch_size: int,
    num_workers: int,
    layer: str | None,
    mask_fraction: float = 0.2,
) -> tuple[dict, pd.DataFrame]:
    """Run MC imputation on the hold-out spatial panel and evaluate against ground truth."""
    _, st_dataset = cellpin.pp.setup_data(
        sc_adata=prepared.adata_sc,
        st_adata=prepared.adata_st,
        layer=layer,
    )
    dl = DataLoader(
        st_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    adata_imputed = model.impute(
        dl,
        obs_adata=prepared.adata_sp,
        mask_fraction=mask_fraction,
        return_norm=False,
        return_int=False,
    )

    imputed_gene_idx = {g: i for i, g in enumerate(adata_imputed.var_names.tolist())}
    valid_test = [g for g in prepared.test_genes if g in imputed_gene_idx]
    if not valid_test:
        raise ValueError("None of the hold-out test_genes appear in the imputed output.")

    y_pred = np.asarray(adata_imputed.X, dtype=float)[
        :, [imputed_gene_idx[g] for g in valid_test]
    ]

    sp_gene_idx = {g: i for i, g in enumerate(prepared.adata_sp.var_names.tolist())}
    X_sp = prepared.adata_sp.X
    if hasattr(X_sp, "toarray"):
        X_sp = X_sp.toarray()
    y_true = np.asarray(X_sp, dtype=float)[
        :, [sp_gene_idx[g] for g in valid_test]
    ]

    return compute_metrics(y_pred, y_true, valid_test)
