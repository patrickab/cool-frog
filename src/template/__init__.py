"""Template package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("template")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
