import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator
from typing_extensions import Self

from ._loaders import (
    load_from_file,
    load_from_json_file,
    load_from_json_string,
    load_from_yaml_file,
    load_from_yaml_string,
)
from .enums import ActionTriggerType, HttpMethod


def _fn_to_tool_definition(fn: Callable[..., Any]) -> "ToolDefinition":
    """Convert a plain callable to a ToolDefinition.

    Uses ``fn.__name__`` as the id, docstring as description, and infers
    the input schema from type hints.
    """
    from superdialog.tools.python_tool import PythonTool

    sig = inspect.signature(fn)
    name = fn.__name__
    description = (fn.__doc__ or "").strip()
    input_schema = PythonTool._infer_schema(sig)
    return ToolDefinition(
        id=name,
        name=name,
        description=description,
        input_schema=input_schema,
        handler_id=name,
        handler=fn,
    )


def _coerce_tool_item(item: Any) -> Any:
    """Coerce a single tools-list item to something Pydantic accepts.

    Handles plain callables and PythonTool / Tool ABC instances.
    Dicts and existing ToolDefinition objects pass through unchanged.
    """
    if isinstance(item, dict):
        return item
    # Tool ABC instance (e.g. PythonTool returned by @tool decorator)
    from superdialog.tools.base import Tool as _ToolABC

    if isinstance(item, _ToolABC):
        return ToolDefinition(
            id=item.id,
            name=item.name,
            description=item.description,
            input_schema=item.input_schema,
            handler_id=item.id,
            handler=getattr(item, "fn", None),
        )
    # Plain callable (regular function or async function)
    if callable(item):
        return _fn_to_tool_definition(item)
    return item


class ToolDefinition(BaseModel):
    """Placeholder — replaced by Tool ABC in superdialog.tools in Task 4.

    Preserves the original ``super.core.voice.dialog_machine.models``
    ``ToolDefinition`` fields and adds discriminator fields
    (``type``, ``url``, ``method``, ``server``) so flow JSONs that
    declare HTTP / MCP tools round-trip cleanly.

    When constructing flows in Python, ``handler`` may hold a callable
    reference directly. It is excluded from serialization so JSON
    round-trips are unaffected.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: str
    name: str
    description: str
    input_schema: dict[str, Any] | None = None
    handler_id: str | None = None
    # Callable reference — set when constructing flows in Python code.
    # Excluded from JSON serialisation; always None for JSON-loaded flows.
    handler: Callable[..., Any] | None = Field(default=None, exclude=True)
    # Discriminator + transport fields (added for the spec-aligned port).
    type: Literal["python", "http", "mcp"] = "python"
    url: str | None = None
    method: str | None = None
    server: str | None = None


class EnvUpdate(BaseModel):
    """Maps an action result field to an environment variable."""

    env_key: str
    result_path: str


class CustomAction(BaseModel):
    model_config = {"populate_by_name": True}

    id: str
    name: str
    description: str = ""
    method: HttpMethod
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    body_template: str | None = Field(default=None, alias="body")
    timeout: int = 30
    store_response_as: str | None = None
    identifier: str | None = None
    env_updates: list[EnvUpdate] = Field(default_factory=list)
    run_once: bool = False
    string_fields: list[str] = Field(default_factory=list)
    condition: str | None = None

    @field_validator("headers", mode="before")
    @classmethod
    def coerce_headers(cls, v: Any) -> dict[str, str]:
        """Accept headers as list of {key, value} dicts or flat dict."""
        if isinstance(v, list):
            return {
                item["key"]: item["value"]
                for item in v
                if isinstance(item, dict) and "key" in item
            }
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("body_template", mode="before")
    @classmethod
    def coerce_body_template(cls, v: Any) -> str | None:
        """Accept body as dict (JSON-serialize) or string."""
        if v is None or v == "":
            return None
        if isinstance(v, dict):
            return json.dumps(v)
        return v


class ActionTrigger(BaseModel):
    model_config = {"populate_by_name": True, "extra": "ignore"}

    trigger_type: ActionTriggerType = Field(alias="trigger")
    action_id: str


class CompletionCriterion(BaseModel):
    """A single criterion for determining if a node's objective is achieved."""

    key: str
    description: str
    required: bool = True


class Edge(BaseModel):
    condition: str
    id: str
    target_node_id: str | None = None
    input_schema: dict[str, Any] | type[BaseModel] | None = None
    actions: list[ActionTrigger] = Field(default_factory=list)
    criteria: list[CompletionCriterion] | None = None
    is_fallback: bool = False
    # Optional machine-evaluable Jinja boolean. When set, the router can decide
    # this edge purely from data (zero LLM) — used for silent/auto-proceed
    # routers whose prose `condition` the deterministic parser can't read.
    # e.g. "hold_result.success == true and hold_result.data.status == 'active'".
    guard: str | None = None

    @field_validator("input_schema", mode="before")
    @classmethod
    def convert_pydantic_to_schema(cls, v):
        if v is None:
            return None

        if isinstance(v, dict):
            return v

        if isinstance(v, type) and issubclass(v, BaseModel):
            schema = v.model_json_schema()
            return {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
                "additionalProperties": schema.get("additionalProperties", False),
            }

        return v


class FlowNode(BaseModel):
    id: str
    name: str
    instruction: str | None = None
    static_text: str | None = None
    is_final: bool = False
    # Explicit node type override ("router", "instruction", "static", "final").
    # When set, classify_node_type() uses this directly without heuristics.
    node_type: str | None = None
    # When True, the runtime advances without waiting for the caller to speak.
    auto_proceed: bool = False
    edges: list[Edge] = Field(default_factory=list)
    actions: list[ActionTrigger] = Field(default_factory=list)
    completion_criteria: list[CompletionCriterion] | None = None
    allow_skip: bool = True
    max_turns: int | None = None
    interruptible: bool = True
    tools: list[ToolDefinition] = Field(default_factory=list)
    # Per-node LangGraph pipeline config (optional)
    langGraph: dict[str, Any] | None = None

    @field_validator("tools", mode="before")
    @classmethod
    def coerce_tools(cls, v: Any) -> list[Any]:
        """Accept plain callables alongside ToolDefinition dicts/objects."""
        if not isinstance(v, list):
            return v
        return [_coerce_tool_item(item) for item in v]

    @field_validator("is_final", mode="before")
    @classmethod
    def coerce_is_final(cls, v):
        if v is None:
            return False
        return v


class ConversationFlow(BaseModel):
    """
    Supported input formats
    -----------------------
    Format      | Source            | Method / config key
    ------------|-------------------|------------------------------------------
    YAML        | file (.yaml/.yml) | from_yaml_file(path)  · livekit_flow_yaml
    YAML        | string            | from_yaml_string(s)   · livekit_flow_yaml_string
    JSON        | file (.json)      | from_json_file(path)  · livekit_flow_json
    JSON        | string            | from_json_string(s)   · flows_json
    auto-detect | file              | from_file(path)       · livekit_flow_file
    any above   | agent config dict | from_config(config)   ← used by handlers
    """

    system_prompt: str
    initial_node: str
    nodes: list[FlowNode]
    actions: list[CustomAction] = Field(default_factory=list)
    environment_variables: dict[str, str] = Field(default_factory=dict)
    global_edges: list[Edge] = Field(default_factory=list)
    agent_gender: str = ""
    agent_language: str = ""
    agent_tone: str = ""
    tools: list[ToolDefinition] = Field(default_factory=list)
    objective: str = ""

    @field_validator("tools", mode="before")
    @classmethod
    def coerce_tools(cls, v: Any) -> list[Any]:
        """Accept plain callables alongside ToolDefinition dicts/objects."""
        if not isinstance(v, list):
            return v
        return [_coerce_tool_item(item) for item in v]

    @field_validator("environment_variables", mode="before")
    @classmethod
    def coerce_env_vars(cls, v: Any) -> dict[str, str]:
        """Accept env_vars as list of {key, value} dicts or flat dict."""
        if isinstance(v, list):
            return {
                item["key"]: item["value"]
                for item in v
                if isinstance(item, dict) and "key" in item
            }
        if isinstance(v, dict):
            return v
        return {}

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> Self:  # type: ignore[override]
        """Override to accept global_actions and env_vars aliases."""
        if isinstance(obj, dict):
            obj = dict(obj)  # shallow copy

            # global_actions → actions
            if "global_actions" in obj and "actions" not in obj:
                obj["actions"] = obj.pop("global_actions")
            elif "global_actions" in obj:
                obj.pop("global_actions")

            # env_vars → environment_variables
            if "env_vars" in obj and "environment_variables" not in obj:
                obj["environment_variables"] = obj.pop("env_vars")
            elif "env_vars" in obj:
                obj.pop("env_vars")

        return super().model_validate(obj, **kwargs)

    # -- YAML ------------------------------------------------------------------

    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> Self:
        """Load from a .yaml / .yml file on disk."""
        return load_from_yaml_file(cls, file_path)

    @classmethod
    def from_yaml_string(cls, yaml_string: str) -> Self:
        """Load from a raw YAML string."""
        return load_from_yaml_string(cls, yaml_string)

    # -- JSON ------------------------------------------------------------------

    @classmethod
    def from_json_file(cls, file_path: Union[str, Path]) -> Self:
        """Load from a .json file on disk."""
        return load_from_json_file(cls, file_path)

    @classmethod
    def from_json_string(cls, json_string: str) -> Self:
        """Load from a raw JSON string."""
        return load_from_json_string(cls, json_string)

    # -- auto-detect -----------------------------------------------------------

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> Self:
        """Load from a file; format is inferred from extension (.yaml/.yml/.json)."""
        return load_from_file(cls, file_path)

    # -- React Flow transformation ---------------------------------------------

    @staticmethod
    def _is_react_flow_format(data: Dict[str, Any]) -> bool:
        """Detect React Flow graph editor JSON format.

        Checks for React Flow UI export signatures:
        - Top-level or config-nested ``initialNodeId``
        - Edges with ``source``/``target`` keys (graph editor format)
        - Nodes with ``type: "flowNode"`` (UI editor node type)
        """
        if "initialNodeId" in data:
            return True
        config = data.get("config")
        if isinstance(config, dict) and "initialNodeId" in config:
            return True
        edges = data.get("edges")
        if isinstance(edges, list) and edges:
            first = edges[0]
            if isinstance(first, dict) and "source" in first and "target" in first:
                return True
        return False

    @staticmethod
    def _normalize_react_flow(data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform React Flow JSON into ConversationFlow dict.

        Handles the React Flow UI editor export format where:
        - Config fields (systemPrompt, initialNodeId) may be nested
          inside a ``config`` wrapper or at top-level.
        - Node data uses camelCase keys (``staticResponse``,
          ``isFinal``, ``responseMode``) instead of snake_case.
        - Edges use ``source``/``target`` instead of inline
          ``target_node_id``.
        """
        logger = logging.getLogger(__name__)
        config = data.get("config", {}) or {}

        # Extract system_prompt — check config wrapper then top-level
        system_prompt = (
            config.get("systemPrompt")
            or data.get("systemPrompt")
            or config.get("system_prompt")
            or data.get("system_prompt")
            or data.get("globalConfig", {}).get("systemPrompt")
            or data.get("globalConfig", {}).get("system_prompt")
            or ""
        )
        if not system_prompt:
            logger.warning(
                "[FLOW] No system_prompt found in React Flow data, using empty string"
            )

        # Map initial_node — check config wrapper then top-level,
        # fall back to first node
        initial_node = (
            config.get("initialNodeId")
            or data.get("initialNodeId")
            or config.get("initial_node_id")
            or data.get("initial_node_id")
            or data.get("initialNode")
            or data.get("initial_node")
            or data.get("startNodeId")
            or ""
        )
        raw_nodes = data.get("nodes", [])
        if not initial_node and raw_nodes:
            initial_node = raw_nodes[0].get("id", "")
            logger.info(
                f"[FLOW] No initialNodeId found, using first node: {initial_node}"
            )

        # Group edges by source node
        edges_by_source: Dict[str, list[Dict[str, Any]]] = {}
        for edge in data.get("edges", []):
            source = edge.get("source", "")
            edge_data = edge.get("data", {}) or {}
            condition = edge_data.get("condition") or edge.get("label") or "default"
            normalized_edge = {
                "id": edge.get("id", ""),
                "condition": condition,
                "target_node_id": edge.get("target"),
            }
            input_schema = (
                edge_data.get("input_schema")
                or edge_data.get("inputSchema")
                or edge.get("input_schema")
                or edge.get("inputSchema")
            )
            if input_schema is not None:
                normalized_edge["input_schema"] = input_schema

            actions = edge_data.get("actions") or edge.get("actions")
            if actions:
                normalized_edge["actions"] = actions

            criteria = edge_data.get("criteria") or edge.get("criteria")
            if criteria:
                normalized_edge["criteria"] = criteria

            is_fallback = edge_data.get("is_fallback")
            if is_fallback is None:
                is_fallback = edge_data.get("isFallback")
            if is_fallback is None:
                is_fallback = edge.get("is_fallback")
            if is_fallback is None:
                is_fallback = edge.get("isFallback")
            if is_fallback is not None:
                normalized_edge["is_fallback"] = is_fallback

            guard = edge_data.get("guard") or edge.get("guard")
            if guard is not None:
                normalized_edge["guard"] = guard

            edges_by_source.setdefault(source, []).append(normalized_edge)

        # Transform nodes — handle both camelCase (UI editor) and
        # snake_case field names
        nodes = []
        for node in data.get("nodes", []):
            node_id = node.get("id", "")
            node_data = node.get("data", {}) or {}

            # Name: try name → label → fallback to node_id
            name = node_data.get("name") or node_data.get("label") or node_id

            # Instruction: try instructions (plural) → instruction
            instruction = (
                node_data.get("instructions") or node_data.get("instruction") or None
            )
            # Empty string means no instruction
            if instruction is not None and not instruction.strip():
                instruction = None

            # Static text: try staticResponse → static_text
            static_text = (
                node_data.get("staticResponse") or node_data.get("static_text") or None
            )
            if static_text is not None and not static_text.strip():
                static_text = None

            # Resolve which field to use based on responseMode
            response_mode = node_data.get("responseMode", "static")
            if response_mode == "llm" and instruction:
                # LLM mode: use instruction, clear static_text
                static_text = None
            elif response_mode == "static" and static_text:
                # Static mode: use static_text, clear instruction
                instruction = None

            # is_final: try isFinal → is_final
            is_final = (
                node_data.get("isFinal")
                if node_data.get("isFinal") is not None
                else node_data.get("is_final", False)
            )

            nodes.append(
                {
                    "id": node_id,
                    "name": name,
                    "instruction": instruction,
                    "static_text": static_text,
                    "is_final": bool(is_final),
                    "edges": edges_by_source.get(node_id, []),
                }
            )

        return {
            "system_prompt": system_prompt,
            "initial_node": initial_node,
            "nodes": nodes,
        }

    # -- from agent config dict ------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional[Self]:
        """
        Load from an agent config dict.  Keys are checked in this priority order:

          livekit_flow_yaml        → from_yaml_file(path)
          livekit_flow_json        → from_json_file(path)
          livekit_flow_yaml_string → from_yaml_string(s)
          flows_json → from_json_string(s)
          livekit_flow_file        → from_file(path)  ← auto-detect by extension

        Returns None if none of the keys are present in the config.
        """
        if path := config.get("livekit_flow_yaml"):
            return cls.from_yaml_file(path)
        if path := config.get("livekit_flow_json"):
            return cls.from_json_file(path)
        if s := config.get("livekit_flow_yaml_string"):
            if cls._is_empty_flow_data(s):
                return None
            return cls.from_yaml_string(s)
        if s := config.get("flows_json"):
            if cls._is_empty_flow_data(s):
                return None
            return cls.from_json_string(s)
        if path := config.get("livekit_flow_file"):
            return cls.from_file(path)
        return None

    @staticmethod
    def _is_empty_flow_data(raw: str) -> bool:
        """Check if a flow string parses to an empty/missing definition."""
        stripped = raw.strip()
        return stripped in ("", "{}", "[]", "null", "~")

    # -- spec-aligned save / load ---------------------------------------------

    def save(self, path: "str | Path") -> None:
        """Serialize this flow to ``path`` as indented JSON."""
        Path(path).write_text(json.dumps(self.model_dump(exclude_unset=True), indent=2))

    @classmethod
    def load(cls, path: "str | Path") -> "ConversationFlow":
        """Load a ConversationFlow from a JSON file at ``path``."""
        return cls.model_validate(json.loads(Path(path).read_text()))
