import pytest

from superdialog.tools import PythonTool, ToolResult


async def hello(name: str) -> dict:
    return {"greeting": f"Hello, {name}"}


def sync_add(a: int, b: int) -> dict:
    return {"sum": a + b}


@pytest.mark.asyncio
async def test_python_tool_from_function() -> None:
    tool = PythonTool.of(hello)
    result = await tool.execute({"name": "World"})
    assert result.data == {"greeting": "Hello, World"}


@pytest.mark.asyncio
async def test_python_tool_infers_schema() -> None:
    tool = PythonTool.of(hello)
    assert tool.input_schema is not None
    assert tool.input_schema["properties"]["name"]["type"] == "string"
    assert "name" in tool.input_schema["required"]


@pytest.mark.asyncio
async def test_python_tool_sync_callable() -> None:
    tool = PythonTool.of(sync_add)
    result = await tool.execute({"a": 1, "b": 2})
    assert result.data == {"sum": 3}
    assert tool.input_schema is not None
    assert tool.input_schema["properties"]["a"]["type"] == "integer"


@pytest.mark.asyncio
async def test_python_tool_returns_toolresult_passthrough() -> None:
    async def fn() -> ToolResult:
        return ToolResult(data={"x": 1}, transition_edge_id="edge-1")

    tool = PythonTool.of(fn)
    result = await tool.execute({})
    assert result.transition_edge_id == "edge-1"
    assert result.data == {"x": 1}


@pytest.mark.asyncio
async def test_python_tool_wraps_scalar_result() -> None:
    async def fn() -> int:
        return 42

    tool = PythonTool.of(fn)
    result = await tool.execute({})
    assert result.data == {"value": 42}


@pytest.mark.asyncio
async def test_python_tool_missing_handler_errors() -> None:
    tool = PythonTool(id="x", name="x", description="", fn=None)
    result = await tool.execute({})
    assert result.error is not None
    assert "no handler bound" in result.error
