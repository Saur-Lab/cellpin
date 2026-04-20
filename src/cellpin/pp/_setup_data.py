from __future__ import annotations

import pathlib
from collections import Counter
from typing import Optional

import anndata as ad
import numpy as np

from cellpin.dataset import scAnnDataset, stAnnDataset


def _get_gene_names(adata: ad.AnnData, gene_symbols: Optional[str]) -> np.ndarray:
    if gene_symbols is not None:
        return adata.var[gene_symbols].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()


def setup_data(
    sc_adata: ad.AnnData,
    st_adata: ad.AnnData,
    gene_symbols: str | None = None,
    layer: str | None = None,
    batch_key: str | None = None,
) -> tuple[scAnnDataset, stAnnDataset]:
    """
      - sc_dataset.panel_genes == st_dataset.panel_genes (same genes, same order)
      - sc_dataset[i]["panel_expr"][k] and st_dataset[i]["full_expr"][k] refer to same gene
    """

    sc_genes = _get_gene_names(sc_adata, gene_symbols)
    st_genes = _get_gene_names(st_adata, gene_symbols)
    sc_set = set(sc_genes.tolist())
    st_set = set(st_genes.tolist())

    # --- Duplicate checks ---
    sc_counts = Counter(sc_genes.tolist())
    sc_dupes = sorted(g for g, n in sc_counts.items() if n > 1)
    if sc_dupes:
        raise ValueError(
            f"sc_adata contains {len(sc_dupes)} duplicate gene name(s): "
            f"{sc_dupes[:10]} ..."
        )
    st_counts = Counter(st_genes.tolist())
    st_dupes = sorted(g for g, n in st_counts.items() if n > 1)
    if st_dupes:
        raise ValueError(
            f"st_adata contains {len(st_dupes)} duplicate gene name(s): "
            f"{st_dupes[:10]} ..."
        )

    # --- Overlap report ---
    missing_from_sc = [g for g in st_genes.tolist() if g not in sc_set]
    if missing_from_sc:
        print(
            f"[cellpin.pp.setup] WARNING: {len(missing_from_sc)} spatial gene(s) not found "
            f"in sc_adata and will be dropped:\n  {missing_from_sc[:10]}"
            + (" ..." if len(missing_from_sc) > 10 else "")
        )

    panel_genes_sp_order = []
    seen = set()
    for g in st_genes.tolist():
        if g in sc_set and g not in seen:
            panel_genes_sp_order.append(g)
            seen.add(g)

    if len(panel_genes_sp_order) == 0:
        raise ValueError("No overlapping genes between spatial and single-cell after matching.")

    # Sort panel genes by sc_adata position order.
    # scAnnDataset uses boolean masking (full_expr[_panel_mask]) which always returns
    # genes in ascending sc_adata index order, regardless of the panel list order.
    # stAnnDataset must match that order so inference gene[i] == training gene[i].
    sc_gene_pos = {g: i for i, g in enumerate(sc_genes.tolist())}
    panel_genes = sorted(panel_genes_sp_order, key=lambda g: sc_gene_pos[g])

    # --- Summary ---
    print(
        f"[cellpin.pp.setup] sc_adata : {sc_adata.n_obs:,} cells × {len(sc_genes):,} genes"
    )
    print(
        f"[cellpin.pp.setup] st_adata : {st_adata.n_obs:,} cells × {len(st_genes):,} spatial genes"
    )
    print(
        f"[cellpin.pp.setup] Panel    : {len(panel_genes):,} genes overlap "
        f"(spatial ∩ scRNA) — these are the genes the model will use as input"
    )
    print(
        f"[cellpin.pp.setup] Imputed  : {len(sc_genes):,} genes (full scRNA gene space)"
    )

    sc_dataset = scAnnDataset(
        adata=sc_adata,
        layer=layer,
        gene_symbols=gene_symbols,
        panel=panel_genes,
        batch_key=batch_key,
    )
    if batch_key is not None and sc_dataset.n_batch > 0:
        print(
            f"[cellpin.pp.setup] Batch key: '{batch_key}' → {sc_dataset.n_batch} categories"
        )

    if hasattr(sc_dataset, "panel_genes"):
        if list(sc_dataset.panel_genes) != list(panel_genes):
            raise ValueError(
                "scAnnDataset panel_genes mismatch vs setup panel_genes. "
                "Ensure scAnnDataset preserves the provided panel order and only drops missing genes."
            )

    # Create stAnnDataset aligned to sc_adata panel order
    st_dataset = stAnnDataset(
        adata=st_adata,
        panel_genes=panel_genes,
        layer=layer,
        gene_symbols=gene_symbols,
    )

    # Final sanity: confirm stAnnDataset gene order matches panel_genes exactly
    if hasattr(st_dataset, "panel_genes"):
        if list(st_dataset.panel_genes) != list(panel_genes):
            raise ValueError(
                "stAnnDataset panel_genes do not match setup panel_genes after construction. "
                "This is a bug — please report it."
            )

    print(
        f"[cellpin.pp.setup] Panel genes sorted to sc_adata position order ✓"
        f" (first 5: {panel_genes[:5]})"
    )

    return sc_dataset, st_dataset


def setup(
    sc_adata: ad.AnnData,
    st_adata: ad.AnnData,
    *,
    layer: str | None = None,
    gene_symbols: str | None = None,
    batch_key: str | None = None,
) -> tuple[scAnnDataset, stAnnDataset]:
    """Set up aligned sc and spatial datasets for CellPin.

    Convenience alias for :func:`setup_data` with keyword-only data arguments.

    Args:
        sc_adata: Single-cell AnnData (full gene panel, used as reference).
        st_adata: Spatial AnnData (panel genes only, used for imputation).
        layer: Layer key to read counts from (``None`` → use ``.X``).
        gene_symbols: ``adata.var`` column with gene names (``None`` → ``var_names``).
        batch_key: ``adata.obs`` column with batch labels for decoder conditioning.
            ``None`` (default) disables batch correction. Must be a column in
            ``sc_adata.obs``; spatial data receives a soft uniform one-hot at
            inference time.

    Returns:
        ``(sc_dataset, st_dataset)`` ready to pass to :class:`~cellpin.models.CellPin`.

    Example::

        sc_dataset, st_dataset = cellpin.pp.setup(sc_adata, st_adata, layer="counts",
                                                   batch_key="Sample_ID")
    """
    return setup_data(
        sc_adata=sc_adata,
        st_adata=st_adata,
        gene_symbols=gene_symbols,
        layer=layer,
        batch_key=batch_key,
    )
