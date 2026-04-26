from importlib.metadata import PackageNotFoundError, version

from cellpin import models, pl, pp
from cellpin.models import CellPin

__all__ = ["CellPin", "models", "pl", "pp"]

try:
    __version__ = version("cellpin")
except PackageNotFoundError:
    __version__ = "unknown"
