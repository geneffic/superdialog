"""FastAPI example: expose a DialogMachine over HTTP.

Run with ``pip install superdialog[fastapi]`` and ``uvicorn examples.fastapi_app:app``.
"""

from __future__ import annotations

from fastapi import FastAPI

from superdialog import DialogMachine
from superdialog.adapters.fastapi import FastAPIRouter
from superdialog.flow.models import ConversationFlow, FlowNode


def _build_dm() -> DialogMachine:
    flow = ConversationFlow(
        id="demo",
        initial_node="welcome",
        nodes=[FlowNode(id="welcome", static_text="Hello!", is_final=True)],
    )
    return DialogMachine(flow=flow, llm="openai/gpt-5.1")


app = FastAPI()
FastAPIRouter(_build_dm()).mount(app, prefix="/dm")
