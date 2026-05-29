from __future__ import annotations

from collections import Counter

import anndata as ad
import numpy as np

from cellpin._sdata_utils import _resolve_sdata
from cellpin.dataset import scAnnDataset, stAnnDataset

_SEP = "=" * 60


def _get_gene_names(adata: ad.AnnData, gene_symbols: str | None) -> np.ndarray:
    if gene_symbols is not None:
        return adata.var[gene_symbols].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()


def setup_data(
    sc_adata: ad.AnnData,
    st_adata: ad.AnnData,
    gene_symbols: str | None = None,
    layer: str | None = None,
    batch_key: str | None = None,
    table_key: str = "table",
) -> tuple[scAnnDataset, stAnnDataset]:
    """
    Prepare aligned single-cell and spatial datasets for CellPin.

    Invariants guaranteed on return:
      - sc_dataset.panel_genes == st_dataset.panel_genes  (same genes, same order)
      - sc_dataset[i]["panel_expr"][k] and st_dataset[i]["full_expr"][k] are the same gene
      - panel_genes order == ascending position order in sc_adata
    """
    print(_SEP)
    print("[cellpin.pp.setup] Setting up CellPin datasets")
    print(_SEP)

    st_adata, _ = _resolve_sdata(st_adata, table_key)

    sc_genes = _get_gene_names(sc_adata, gene_symbols)
    st_genes = _get_gene_names(st_adata, gene_symbols)
    sc_set = set(sc_genes.tolist())

    print(f"[cellpin.pp.setup] sc_adata : {sc_adata.n_obs:>8,} cells  × {len(sc_genes):>8,} genes")
    print(f"[cellpin.pp.setup] st_adata : {st_adata.n_obs:>8,} cells  × {len(st_genes):>8,} spatial genes")
    if gene_symbols is not None:
        print(f"[cellpin.pp.setup] Gene IDs taken from var['{gene_symbols}']")
    if layer is not None:
        print(f"[cellpin.pp.setup] Expression read from layer='{layer}'")

    # check for duplicate gene names
    sc_counts = Counter(sc_genes.tolist())
    sc_dupes = sorted(g for g, n in sc_counts.items() if n > 1)
    if sc_dupes:
        raise ValueError(
            f"sc_adata contains {len(sc_dupes)} duplicate gene name(s): "
            f"{sc_dupes[:10]}" + (" ..." if len(sc_dupes) > 10 else "")
        )
    st_counts = Counter(st_genes.tolist())
    st_dupes = sorted(g for g, n in st_counts.items() if n > 1)
    if st_dupes:
        raise ValueError(
            f"st_adata contains {len(st_dupes)} duplicate gene name(s): "
            f"{st_dupes[:10]}" + (" ..." if len(st_dupes) > 10 else "")
        )
    missing_from_sc = [g for g in st_genes.tolist() if g not in sc_set]
    if missing_from_sc:
        print(
            f"[cellpin.pp.setup] WARNING: {len(missing_from_sc)} spatial gene(s) not found "
            f"in sc_adata — will be dropped:\n"
            f"  {missing_from_sc[:10]}" + (" ..." if len(missing_from_sc) > 10 else "")
        )

    panel_genes_sp_order: list[str] = []
    seen: set[str] = set()
    for g in st_genes.tolist():
        if g in sc_set and g not in seen:
            panel_genes_sp_order.append(g)
            seen.add(g)

    if len(panel_genes_sp_order) == 0:
        raise ValueError(
            "No overlapping genes between spatial and single-cell after matching.\n"
            f"  sc_adata genes (first 10): {sc_genes[:10].tolist()}\n"
            f"  st_adata genes (first 10): {st_genes[:10].tolist()}"
        )

    # Sort panel genes to sc_adata position order.
    # scAnnDataset uses boolean masking (full_expr[_panel_mask]) which always returns
    # genes in ascending sc_adata index order, regardless of the panel list order.
    # stAnnDataset must match that order so inference gene[i] == training gene[i].
    sc_gene_pos: dict[str, int] = {g: i for i, g in enumerate(sc_genes.tolist())}
    panel_genes = sorted(panel_genes_sp_order, key=lambda g: sc_gene_pos[g])

    n_panel = len(panel_genes)
    n_imputed_only = len(sc_genes) - n_panel
    overlap_pct = 100.0 * n_panel / len(st_genes)

    print(f"[cellpin.pp.setup] Panel    : {n_panel:,} genes overlap ({overlap_pct:.1f}% of spatial genes retained)")
    print(
        f"[cellpin.pp.setup] Imputed  : {len(sc_genes):,} genes total in sc space "
        f"({n_imputed_only:,} genes to impute, not in panel)"
    )

    if n_panel < 10:
        print(
            f"[cellpin.pp.setup] WARNING: only {n_panel} panel gene(s) — this is very small. "
            "Check that gene names in sc_adata and st_adata match (same symbols, same case)."
        )

    sc_dataset = scAnnDataset(
        adata=sc_adata,
        layer=layer,
        gene_symbols=gene_symbols,
        panel=panel_genes,
        batch_key=batch_key,
    )

    if batch_key is not None and sc_dataset.n_batch == 0:
        print(
            f"[cellpin.pp.setup] WARNING: batch_key='{batch_key}' not found in sc_adata.obs "
            "— batch conditioning disabled."
        )

    # 1. Gene names must be preserved unchanged
    if list(sc_dataset.gene_names) != list(sc_genes):
        raise ValueError(
            "scAnnDataset.gene_names does not match sc_adata gene names. "
            "This is a bug — please report it.\n"
            f"  Expected (first 5): {list(sc_genes[:5])}\n"
            f"  Got      (first 5): {list(sc_dataset.gene_names[:5])}"
        )

    # 2. panel_genes list must equal our sorted panel_genes exactly
    if list(sc_dataset.panel_genes) != list(panel_genes):
        raise ValueError(
            "scAnnDataset.panel_genes does not match setup panel_genes.\n"
            f"  Expected (first 5): {panel_genes[:5]}\n"
            f"  Got      (first 5): {list(sc_dataset.panel_genes)[:5]}"
        )

    # 3. panel_idx must be strictly ascending (proves boolean-mask order equals panel_genes order)
    if len(sc_dataset.panel_idx) > 1:
        if not np.all(np.diff(sc_dataset.panel_idx) > 0):
            raise ValueError(
                "scAnnDataset.panel_idx is not strictly ascending. "
                "The boolean mask will return genes in a different order than panel_genes. "
                "This is a bug — please report it.\n"
                f"  panel_idx: {sc_dataset.panel_idx.tolist()}"
            )

    # 4. panel_idx values must map to the correct gene names
    for k, (idx, gene) in enumerate(zip(sc_dataset.panel_idx.tolist(), panel_genes, strict=False)):
        actual = sc_dataset.gene_names[idx]
        if actual != gene:
            raise ValueError(
                f"scAnnDataset.panel_idx[{k}]={idx} points to gene '{actual}' "
                f"but panel_genes[{k}]='{gene}'. "
                "This is a bug — please report it."
            )

    # Build stAnnDataset aligned to sc_adata panel order                  #
    st_dataset = stAnnDataset(
        adata=st_adata,
        panel_genes=panel_genes,
        layer=layer,
        gene_symbols=gene_symbols,
    )

    # 5. stAnnDataset must preserve panel_genes order exactly
    if list(st_dataset.panel_genes) != list(panel_genes):
        raise ValueError(
            "stAnnDataset.panel_genes does not match setup panel_genes after construction.\n"
            f"  Expected (first 5): {panel_genes[:5]}\n"
            f"  Got      (first 5): {list(st_dataset.panel_genes)[:5]}"
        )

    print(
        f"[cellpin.pp.setup] sc_dataset: {len(sc_dataset):,} cells, "
        f"{len(sc_dataset.gene_names):,} genes total, "
        f"{len(sc_dataset.panel_genes):,} panel genes"
    )
    print(f"[cellpin.pp.setup] st_dataset: {len(st_dataset):,} cells, {len(st_dataset.panel_genes):,} panel genes")

    return sc_dataset, st_dataset


