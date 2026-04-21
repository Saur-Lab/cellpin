from __future__ import annotations

import pathlib
import urllib.request

import anndata as ad

_BASE_URL = "https://raw.githubusercontent.com/Saur-Lab/cellpin-data/main/datasets"
_SC_EXAMPLE_NAME = "sc_example.h5ad"
_SP_EXAMPLE_NAME = "sp_example.h5ad"


def _default_cache_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".cache" / "cellpin" / "datasets"


def _download_if_needed(
    file_name: str,
    *,
    cache_dir: str | pathlib.Path | None = None,
    force_download: bool = False,
) -> pathlib.Path:
    destination_dir = pathlib.Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / file_name
    if force_download or not destination.exists():
        url = f"{_BASE_URL}/{file_name}"
        urllib.request.urlretrieve(url, destination)

    return destination


def load_sc_example(
    *,
    cache_dir: str | pathlib.Path | None = None,
    force_download: bool = False,
) -> ad.AnnData:
    """Load the single-cell example dataset used by CellPin tutorials.

    Downloads ``sc_example.h5ad`` from the CellPin data repository into a
    local cache directory on first use and then loads it with ``anndata``.

    Args:
        cache_dir: Optional custom cache directory for downloaded files.
            If ``None``, defaults to ``~/.cache/cellpin/datasets``.
        force_download: If ``True``, always re-download the file even when a
            cached copy exists.

    Returns:
        Loaded single-cell example ``AnnData``.
    """
    path = _download_if_needed(
        _SC_EXAMPLE_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    return ad.read_h5ad(path)


def load_sp_example(
    *,
    cache_dir: str | pathlib.Path | None = None,
    force_download: bool = False,
) -> ad.AnnData:
    """Load the spatial example dataset used by CellPin tutorials.

    Downloads ``sp_example.h5ad`` from the CellPin data repository into a
    local cache directory on first use and then loads it with ``anndata``.

    Args:
        cache_dir: Optional custom cache directory for downloaded files.
            If ``None``, defaults to ``~/.cache/cellpin/datasets``.
        force_download: If ``True``, always re-download the file even when a
            cached copy exists.

    Returns:
        Loaded spatial example ``AnnData``.
    """
    path = _download_if_needed(
        _SP_EXAMPLE_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    return ad.read_h5ad(path)
