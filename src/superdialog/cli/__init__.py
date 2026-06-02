"""superdialog command-line interface.

Entry point is :func:`main`; the ``superdialog`` console script declared in
``pyproject.toml`` resolves to ``superdialog.cli.main:main``.
"""

from .main import main

__all__ = ["main"]
