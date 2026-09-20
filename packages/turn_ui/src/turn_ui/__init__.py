"""Browser UI for the live human-side end-of-turn runtime."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

try:
    __version__ = package_version("turn-ui")
except PackageNotFoundError:
    __version__ = "0.1.0"
"""Installed `turn-ui` version, or `0.1.0` from a source tree."""

__all__ = ["__version__"]
