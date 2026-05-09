from importlib.metadata import PackageNotFoundError, version

from cellpin import models, pl, pp, tl
from cellpin.models import CellPin

__all__ = ["CellPin", "models", "pl", "pp", "tl"]

try:
    __version__ = version("cellpin")
except PackageNotFoundError:
    __version__ = "unknown"
