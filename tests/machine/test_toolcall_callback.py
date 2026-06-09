"""Tests for _on_llm_complete callback and token capture."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from superdialog.machine.adapters.toolcall_adapter import LLMCallData, ToolCallAdapter


def _make_adapter() -> ToolCallAdapter:
    return ToolCallAdapter(model_id="openai/gpt-4.1-mini", system_prompt="")


def test_llm_call_data_fields():
    data = LLMCallData(
        node_id="greet",
        model="gpt-4.1-mini",
        call_type="routing",
        latency_ms=1200.5,
        tokens_in=450,
        tokens_out=12,
        prompt_messages=[{"role": "user", "content": "hi"}],
        response_json={"tool_call": "edge_greet"},
        edge_id="edge_greet",
    )
    assert data.call_type == "routing"
    assert data.tokens_in == 450


@pytest.mark.anyio
async def test_on_llm_complete_callback_fires_on_generate_via_llm():
    adapter = _make_adapter()
    received: list[LLMCallData] = []

    async def _cb(d: LLMCallData) -> None:
        received.append(d)

    adapter._on_llm_complete = _cb

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "Hello!"
    fake_response.usage = MagicMock()
    fake_response.usage.prompt_tokens = 100
    fake_response.usage.completion_tokens = 20

    with patch(
        "superdialog.machine.adapters.toolcall_adapter._make_openai_client"
    ) as mock_client_fn:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)
        mock_client.base_url = "https://api.openai.com"
        mock_client_fn.return_value = mock_client

        result = await adapter._generate_via_llm("Say hello", [], node_id="greet")

    assert len(received) == 1
    assert received[0].call_type == "generate_reply"
    assert received[0].tokens_in == 100
    assert received[0].tokens_out == 20
    assert received[0].node_id == "greet"