#!/usr/bin/env python3
"""Preflight checks for cellpin inputs.

Catches the mistakes that silently ruin cellpin results *before* training:
normalized data passed as counts, gene-identifier mismatches, tiny references,
missing batch/label/area columns, label classes too small for label transfer.

CLI
---
    python check_inputs.py --sc reference.h5ad --sp spatial.h5ad \
        --layer counts --batch-key batch --label-key cell_type

    --sp may also be a SpatialData .zarr directory (uses --table-key, default "table").

Python
------
    from check_inputs import run_checks
    ok = run_checks(sc_adata, sp_adata, layer="counts", batch_key="batch", label_key="cell_type")

Exit code / return value: 1 / False if any FAIL, else 0 / True.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse

MIN_REF_CELLS = 20_000
MAX_REF_GENES_TARGETED = 8_000
N_SAMPLE_ROWS = 2_000
ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d{6,}")
CONTROL_RE = re.compile(r"^(NegControl|BLANK|Unassigned|DeprecatedCodeword|antisense|NegPrb|SystemControl)", re.I)


class Report:
    """Collects and prints PASS/WARN/FAIL lines."""

    def __init__(self) -> None:
        self.n_fail = 0
        self.n_warn = 0

    def ok(self, msg: str) -> None:
        """Print a OK line."""
        print(f"  [PASS] {msg}")

    def warn(self, msg: str) -> None:
        """Print a WARN line."""
        self.n_warn += 1
        print(f"  [WARN] {msg}")

    def fail(self, msg: str) -> None:
        """Print a FAIL line."""
        self.n_fail += 1
        print(f"  [FAIL] {msg}")

    def info(self, msg: str) -> None:
        """Print a INFO line."""
        print(f"         {msg}")


# --------------------------------------------------------------------------- helpers


def _sample_rows(X, n: int = N_SAMPLE_ROWS, seed: int = 0) -> np.ndarray:
    n_obs = X.shape[0]
    idx = np.sort(np.random.default_rng(seed).choice(n_obs, size=min(n, n_obs), replace=False))
    sub = X[idx]
    if sparse.issparse(sub):
        sub = sub.toarray()
    return np.asarray(sub, dtype=np.float64)


def _count_stats(X) -> dict:
    """Return integer/negativity/NaN stats on a row sample of X."""
    v = _sample_rows(X)
    nz = v[v != 0]
    return {
        "has_nan": bool(np.isnan(v).any()),
        "has_neg": bool((v < 0).any()),
        "is_int": bool(nz.size == 0 or np.allclose(nz, np.round(nz))),
        "max": float(np.nanmax(v)) if v.size else 0.0,
        "row_sum_median": float(np.median(v.sum(axis=1))) if v.size else 0.0,
    }


def _looks_raw(stats: dict) -> bool:
    return stats["is_int"] and not stats["has_neg"] and not stats["has_nan"]


def _get_matrix(adata, layer):
    return adata.X if layer is None else adata.layers[layer]


def _raw_count_candidates(adata) -> list[str]:
    """List places in adata that look like raw counts."""
    found = []
    cands = [(".X", adata.X)] + [(f'layers["{k}"]', v) for k, v in adata.layers.items()]
    if adata.raw is not None:
        cands.append((".raw.X", adata.raw.X))
    for name, mat in cands:
        try:
            if mat is not None and _looks_raw(_count_stats(mat)):
                found.append(name)
        except Exception:  # noqa: BLE001
            pass
    return found


def _gene_names(adata, gene_symbols):
    if gene_symbols is None:
        return pd.Index(adata.var_names.astype(str))
    return pd.Index(adata.var[gene_symbols].astype(str))


def _best_var_column(src, target_genes: set[str]) -> tuple[str | None, int]:
    best, best_n = None, 0
    for col in src.var.columns:
        dt = src.var[col].dtype
        if pd.api.types.is_object_dtype(dt) or isinstance(dt, pd.CategoricalDtype) or pd.api.types.is_string_dtype(dt):
            n = len(set(src.var[col].astype(str)) & target_genes)
            if n > best_n:
                best, best_n = col, n
    return best, best_n


# --------------------------------------------------------------------------- checks


def _check_counts(rep: Report, adata, layer, who: str) -> None:
    where = ".X" if layer is None else f'layers["{layer}"]'
    if layer is not None and layer not in adata.layers:
        rep.fail(f"{who}: layer {layer!r} not found. Available layers: {list(adata.layers.keys())}")
        cands = _raw_count_candidates(adata)
        if cands:
            rep.info(f"Raw-count-like data found at: {', '.join(cands)}. Copy it to {where} in BOTH objects.")
        return
    st = _count_stats(_get_matrix(adata, layer))
    if st["has_nan"]:
        rep.fail(f"{who}: {where} contains NaN values.")
    if st["has_neg"]:
        rep.fail(f"{who}: {where} contains negative values (scaled data?). cellpin needs raw counts.")
    if not st["is_int"]:
        rep.fail(f"{who}: {where} is not integer-valued (max={st['max']:.2f}). Looks normalized/log-transformed.")
        cands = [c for c in _raw_count_candidates(adata) if c != where]
        if cands:
            rep.info(f"Raw-count-like data found at: {', '.join(cands)}. Use that (same layer name in both objects).")
        else:
            rep.info("No raw counts found in this object. Get the raw-count version of the dataset.")
    elif not st["has_neg"] and not st["has_nan"]:
        rep.ok(
            f"{who}: {where} looks like raw integer counts (max={st['max']:.0f}, median counts/cell={st['row_sum_median']:.0f})."
        )
        if st["max"] <= 1:
            rep.warn(f"{who}: max value is {st['max']:.0f}; binarized data? cellpin expects raw counts.")


def _check_genes(rep: Report, sc_adata, sp_adata, gene_symbols, platform_wt: bool) -> None:
    for who, a in (("reference", sc_adata), ("spatial", sp_adata)):
        if gene_symbols is not None and gene_symbols not in a.var.columns:
            rep.fail(f"{who}: gene_symbols column {gene_symbols!r} not in var (gene_symbols applies to BOTH objects).")
            return

    sc_g = _gene_names(sc_adata, gene_symbols)
    sp_g = _gene_names(sp_adata, gene_symbols)

    for who, g in (("reference", sc_g), ("spatial", sp_g)):
        d = g[g.duplicated()].unique()
        if len(d):
            rep.fail(f"{who}: {len(d)} duplicate gene names (e.g. {list(d[:5])}). Run var_names_make_unique().")

    ctrl = [x for x in sp_g if CONTROL_RE.match(x)]
    if ctrl:
        rep.warn(f"spatial: {len(ctrl)} control/blank probes (e.g. {ctrl[:3]}). Consider dropping them.")

    sp_real = [x for x in sp_g if not CONTROL_RE.match(x)]
    overlap = set(sp_real) & set(sc_g)
    frac = len(overlap) / max(len(sp_real), 1)
    msg = f"gene overlap: {len(overlap):,} of {len(sp_real):,} spatial genes found in reference ({frac:.0%})."
    if len(overlap) < 10 or frac < 0.5:
        rep.fail(msg)
        sc_ens = np.mean([bool(ENSEMBL_RE.match(x)) for x in sc_g[:500]]) > 0.5
        sp_ens = np.mean([bool(ENSEMBL_RE.match(x)) for x in sp_g[:500]]) > 0.5
        if sc_ens != sp_ens:
            rep.info(
                f"Ensembl IDs vs symbols: reference uses {'Ensembl' if sc_ens else 'symbols'}, "
                f"spatial uses {'Ensembl' if sp_ens else 'symbols'}."
            )
        for src, tgt, name in ((sc_adata, set(sp_real), "reference"), (sp_adata, set(sc_g), "spatial")):
            col, n = _best_var_column(src, tgt)
            if col is not None and n > len(overlap):
                rep.info(
                    f"Using {name}.var[{col!r}] as gene names would give {n:,} overlapping genes. "
                    f"e.g. {name}.var_names = {name}.var[{col!r}].astype(str); var_names_make_unique()"
                )
        up = {x.upper() for x in sp_real} & {x.upper() for x in sc_g}
        if len(up) > 1.5 * max(len(overlap), 1):
            rep.info(
                f"Case-insensitive overlap is {len(up):,}. Mouse vs human mismatch? Check species; do not just uppercase."
            )
        strip = {x.split(".")[0] for x in sp_real} & {x.split(".")[0] for x in sc_g}
        if len(strip) > 1.5 * max(len(overlap), 1):
            rep.info("Stripping Ensembl version suffixes ('.12') would fix the overlap.")
    elif frac < 0.9:
        rep.warn(msg + " Missing genes are dropped. Check their names if they matter.")
    else:
        rep.ok(msg)

    n_imp = len(set(sc_g) - set(sp_g))
    rep.info(f"reference genes: {len(sc_g):,} | panel (overlap): {len(overlap):,} | imputed-only: {n_imp:,}")
    if not platform_wt and len(sc_g) > MAX_REF_GENES_TARGETED:
        rep.warn(
            f"reference has {len(sc_g):,} genes. Recommended: panel genes + 1,000-2,000 HVGs "
            "(targeted panels) or ~4,000 HVGs (whole transcriptome)."
        )


def _check_reference(rep: Report, sc_adata, batch_key, label_key) -> None:
    n = sc_adata.n_obs
    if n < MIN_REF_CELLS:
        rep.warn(f"reference has {n:,} cells (< {MIN_REF_CELLS:,} recommended).")
    else:
        rep.ok(f"reference has {n:,} cells.")
    if n > 30_000:
        rep.info("reference has > 30k cells -> use fit(..., batch_size=512) (1024 for very large atlases).")

    if batch_key is None:
        cands = [c for c in sc_adata.obs.columns if re.search(r"batch|donor|sample|dataset|study|assay", c, re.I)]
        if cands:
            rep.warn(
                f"no batch_key given, but reference has candidate columns {cands[:6]}. "
                "Pass one if it is a multi-study/donor atlas."
            )
    elif batch_key not in sc_adata.obs.columns:
        rep.fail(f"batch_key {batch_key!r} not in reference obs. Columns: {list(sc_adata.obs.columns)[:20]}")
    else:
        vc = sc_adata.obs[batch_key].astype(str).value_counts()
        rep.ok(f"batch_key {batch_key!r}: {len(vc)} batches (smallest: {vc.min():,} cells).")
        if len(vc) == 1:
            rep.warn("batch_key has a single category, so it has no effect.")
        if len(vc) > 0.1 * n:
            rep.warn("batch_key has very many categories. Is this really a batch column (not a cell ID)?")

    if label_key is not None:
        if label_key not in sc_adata.obs.columns:
            rep.fail(f"label_key {label_key!r} not in reference obs. Columns: {list(sc_adata.obs.columns)[:20]}")
        else:
            vc = sc_adata.obs[label_key].astype(str).value_counts()
            tiny = vc[vc < 2]
            small = vc[(vc >= 2) & (vc < 20)]
            if len(tiny):
                rep.fail(
                    f"label_key {label_key!r}: {len(tiny)} class(es) with < 2 cells {list(tiny.index[:5])}. "
                    "label_transfer will crash. Drop/merge them."
                )
            if len(small):
                rep.warn(
                    f"label_key {label_key!r}: {len(small)} class(es) with < 20 cells {list(small.index[:5])}. "
                    "Labels will be unreliable."
                )
            if not len(tiny):
                rep.ok(f"label_key {label_key!r}: {len(vc)} classes.")
            if sc_adata.obs[label_key].isna().any():
                rep.warn(
                    f"label_key {label_key!r} has {int(sc_adata.obs[label_key].isna().sum()):,} NaN labels "
                    "(they become the class 'nan'). Drop those cells or fill them."
                )


def _check_spatial(rep: Report, sp_adata, layer, area_key) -> None:
    rep.ok(f"spatial has {sp_adata.n_obs:,} cells x {sp_adata.n_vars:,} genes.")
    if area_key is not None:
        if area_key not in sp_adata.obs.columns:
            rep.warn(f"area_key {area_key!r} not in spatial obs. Use area_key=None in impute().")
        else:
            a = pd.to_numeric(sp_adata.obs[area_key], errors="coerce")
            bad = int((~(a > 0)).sum())
            if bad:
                rep.fail(f"{bad:,} cells have {area_key} <= 0 or NaN. Drop them or use area_key=None.")
            else:
                rep.ok(f"area_key {area_key!r} present and positive.")
    if "spatial" not in sp_adata.obsm:
        rep.warn('no obsm["spatial"] coordinates. Spatial plots will not work (cellpin itself does not need them).')

    if layer is None or layer in sp_adata.layers:
        X = _get_matrix(sp_adata, layer)
        n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
        low = int((n_genes < 10).sum())
        zero = int((n_genes == 0).sum())
        if zero:
            rep.warn(f"{zero:,} spatial cells have zero counts. Filter them (sc.pp.filter_cells(min_genes=10)).")
        elif low > 0.05 * sp_adata.n_obs:
            rep.warn(
                f"{low:,} spatial cells ({low / sp_adata.n_obs:.0%}) have < 10 detected genes. Consider filtering."
            )


def _check_gpu(rep: Report) -> None:
    try:
        import torch
    except ImportError:
        rep.warn("torch not importable here. Is cellpin installed in this environment?")
        return
    if torch.cuda.is_available():
        rep.ok(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        rep.warn("no CUDA GPU detected. Training will be slow on CPU.")


def run_checks(
    sc_adata,
    sp_adata,
    layer: str | None = None,
    batch_key: str | None = None,
    label_key: str | None = None,
    area_key: str | None = "cell_area",
    gene_symbols: str | None = None,
    table_key: str = "table",
    whole_transcriptome: bool = False,
    check_gpu: bool = True,
) -> bool:
    """Run all preflight checks. Returns True if there are no FAILs."""
    if hasattr(sp_adata, "tables"):  # SpatialData
        sp_adata = sp_adata.tables[table_key]

    rep = Report()
    print("=" * 72)
    print("cellpin preflight")
    print("=" * 72)
    print("\n[1] Raw counts")
    _check_counts(rep, sc_adata, layer, "reference")
    _check_counts(rep, sp_adata, layer, "spatial")
    print("\n[2] Genes")
    _check_genes(rep, sc_adata, sp_adata, gene_symbols, whole_transcriptome)
    print("\n[3] Reference")
    _check_reference(rep, sc_adata, batch_key, label_key)
    print("\n[4] Spatial")
    _check_spatial(rep, sp_adata, layer, area_key)
    if check_gpu:
        print("\n[5] Hardware")
        _check_gpu(rep)
    print("\n" + "=" * 72)
    if rep.n_fail:
        print(f"RESULT: {rep.n_fail} FAIL, {rep.n_warn} WARN -> fix FAILs before training.")
    else:
        print(f"RESULT: OK ({rep.n_warn} WARN). Review warnings, then proceed.")
    print("=" * 72)
    return rep.n_fail == 0


def _read_any(path: str, table_key: str):
    import anndata as ad

    if path.rstrip("/").endswith(".zarr"):
        try:
            import spatialdata
        except ImportError:
            sys.exit('Reading .zarr needs SpatialData: pip install "cellpin[spatial]"')
        return spatialdata.read_zarr(path).tables[table_key]
    return ad.read_h5ad(path)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description="Preflight checks for cellpin inputs.")
    p.add_argument("--sc", required=True, help="Reference scRNA-seq .h5ad")
    p.add_argument("--sp", required=True, help="Spatial .h5ad or SpatialData .zarr")
    p.add_argument("--layer", default=None, help="Layer with raw counts in BOTH objects (default: .X)")
    p.add_argument("--batch-key", default=None)
    p.add_argument("--label-key", default=None)
    p.add_argument("--area-key", default="cell_area", help='Spatial obs column with cell area ("none" to skip)')
    p.add_argument("--gene-symbols", default=None, help="var column used as gene IDs in BOTH objects")
    p.add_argument("--table-key", default="table")
    p.add_argument(
        "--whole-transcriptome", action="store_true", help="Spatial data is whole-transcriptome (e.g. Atera)"
    )
    p.add_argument("--no-gpu-check", action="store_true")
    a = p.parse_args()

    print(f"Loading reference: {a.sc}")
    sc_adata = _read_any(a.sc, a.table_key)
    print(f"Loading spatial  : {a.sp}")
    sp_adata = _read_any(a.sp, a.table_key)
    ok = run_checks(
        sc_adata,
        sp_adata,
        layer=a.layer,
        batch_key=a.batch_key,
        label_key=a.label_key,
        area_key=None if str(a.area_key).lower() == "none" else a.area_key,
        gene_symbols=a.gene_symbols,
        table_key=a.table_key,
        whole_transcriptome=a.whole_transcriptome,
        check_gpu=not a.no_gpu_check,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
