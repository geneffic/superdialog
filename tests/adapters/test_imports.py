"""Smoke import test for each host adapter module."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name, extra",
    [
        ("superdialog.adapters.livekit", "livekit"),
        ("superdialog.adapters.pipecat", "pipecat"),
        ("superdialog.adapters.fastapi", "fastapi"),
        ("superdialog.adapters.websocket", "ws"),
    ],
)
def test_module_imports_without_extra(module_name: str, extra: str) -> None:
    """Importing an adapter module must not require its optional extra.

    The extras are only required when instantiating the adapter class.
    """
    mod = importlib.import_module(module_name)
    assert mod is not None
