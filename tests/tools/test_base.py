import pytest

from superdialog.tools import HttpTool, MCPTool, PythonTool
from superdialog.tools.base import Tool


def my_handler(x: int) -> dict:
    return {"x": x}


def test_from_dict_python_with_registry() -> None:
    spec = {
        "type": "python",
        "id": "do_thing",
        "description": "does",
        "handler_id": "my_handler",
        "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }
    tool = Tool.from_dict(spec, handler_registry={"my_handler": my_handler})
    assert isinstance(tool, PythonTool)
    assert tool.id == "do_thing"
    assert tool.fn is my_handler


def test_from_dict_http() -> None:
    spec = {
        "type": "http",
        "id": "lookup",
        "description": "",
        "url": "https://e.com/api",
        "method": "GET",
    }
    tool = Tool.from_dict(spec)
    assert isinstance(tool, HttpTool)
    assert tool.method == "GET"
    assert tool.url == "https://e.com/api"


def test_from_dict_mcp() -> None:
    spec = {"type": "mcp", "id": "search", "description": "", "server": "https://m"}
    tool = Tool.from_dict(spec)
    assert isinstance(tool, MCPTool)
    assert tool.server == "https://m"


def test_from_dict_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tool type"):
        Tool.from_dict({"type": "wat", "id": "x"})


def test_to_openai_function() -> None:
    tool = HttpTool(
        id="lookup",
        name="lookup",
        description="lookup desc",
        url="https://e.com",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    spec = tool.to_openai_function()
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "lookup"
    assert spec["function"]["description"] == "lookup desc"
    assert spec["function"]["parameters"]["properties"]["q"]["type"] == "string"
