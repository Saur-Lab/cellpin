from __future__ import annotations

import anndata as ad


def _resolve_sdata(
    obj: ad.AnnData | None,
    table_key: str = "table",
) -> tuple[ad.AnnData | None, object]:
    """Return (adata, sdata_or_None).

    If *obj* is a :class:`spatialdata.SpatialData`, extract ``obj.tables[table_key]``
    and return the sdata for later re-wrapping.  If *obj* is already an
    :class:`anndata.AnnData` or ``None``, pass it through unchanged.
    """
    if obj is None:
        return None, None
    try:
        import spatialdata as sd  # soft dependency
    except ImportError:
        return obj, None
    if isinstance(obj, sd.SpatialData):
        if table_key not in obj.tables:
            raise ValueError(f"SpatialData has no table '{table_key}'. Available tables: {list(obj.tables.keys())}")
        return obj.tables[table_key], obj
    return obj, None
