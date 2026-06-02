"""ToolCallAdapter -- simulates SimpleFlowAgent's tool-call routing.

Uses OpenAI function_calling to pick edges instead of CriteriaJudge.
Same decision path as production (SimpleFlowAgent) without LiveKit.

Usage::

    adapter = ToolCallAdapter(model_id="gpt-4o-mini", system_prompt=prompt)
    machine = await DialogStateMachine.from_flow(flow, adapter)
    result = await machine.process_turn("I want to book an appointment")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

from jinja2 import BaseLoader, ChainableUndefined, Environment

from superdialog.machine.composer import _LANG_MARKER_RE
from superdialog.machine.composer import extract_speech_text as _extract_speech_text
from superdialog.machine.composer import get_time_context as _get_time_context
from superdialog.machine.composer import process_text as _process_text
from superdialog.machine.composer import resolve_language as _resolve_lang
from superdialog.machine.models import CriteriaResult, ToolDescriptor

if TYPE_CHECKING:
    from superdialog.flow.models import CustomAction, FlowNode

logger = logging.getLogger(__name__)


def _is_zero_speech_step(rendered_instruction: str) -> bool:
    """Return True if the rendered instruction mandates zero speech at the current step.

    Conditions (both must hold):
    1. The instruction contains the explicit zero-speech directive.
    2. All ``current_*=VALUE`` Jinja2 template variables have non-empty values
       (indicates STEP 4 / auto-proceed conditions are met — all required fields set).

    This is a deterministic code-level gate that never relies on the LLM to
    voluntarily output an empty string (which is unreliable).
    """
    _ZERO_SPEECH_DIRECTIVE = "ZERO speech — no words before the tool call"
    if _ZERO_SPEECH_DIRECTIVE not in rendered_instruction:
        return False
    # Find every current_*=VALUE pattern; if any value is empty, STEP 4 does not apply.
    found_any = False
    for m in re.finditer(r"current_\w+=(\S*)", rendered_instruction):
        found_any = True
        if not m.group(1):  # empty value → required field missing
            return False
    return found_any


def _extract_agent_says(instruction: str) -> str | None:
    """Extract 'Agent says: <text>', stripping parenthetical stage directions."""
    if not instruction or not instruction.lstrip().startswith("Agent says:"):
        return None
    first_line = instruction.split("\n")[0]
    text = first_line[len("Agent says:"):].strip()
    # Strip (stage directions like this) — not meant for caller
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    return text if text else None


def _strip_provider_prefix(model_id: str) -> str:
    """Strip 'openai/' prefix so OpenAI client gets a bare model name."""
    if "/" in model_id:
        return model_id.split("/", 1)[1]
    return model_id


def _coerce_numeric_strings(d: dict, skip_fields: set[str]) -> dict:
    """Coerce string values that look like ints/floats — Jinja2 renders everything as str."""
    for k, v in d.items():
        if k in skip_fields:
            continue
        if isinstance(v, str):
            try:
                d[k] = int(v)
                continue
            except ValueError:
                pass
            try:
                d[k] = float(v)
            except ValueError:
                pass
        elif isinstance(v, dict):
            _coerce_numeric_strings(v, skip_fields)
    return d


def _descriptors_to_openai_tools(
    descriptors: list[ToolDescriptor],
) -> list[dict[str, Any]]:
    """Convert ToolDescriptors to OpenAI function-calling tool schemas."""
    tools: list[dict[str, Any]] = []
    for desc in descriptors:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": desc.id,
                    "description": desc.description,
                    "parameters": desc.input_schema
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return tools


class ToolCallAdapter:
    """Runtime adapter that uses LLM tool-calling to route edges.

    Mirrors SimpleFlowAgent's instruction construction and presents
    edges as OpenAI function tools. The LLM picks a tool_call instead
    of returning structured JSON via CriteriaJudge.

    Also executes HTTP actions (on_enter/on_exit webhooks) identically
    to LLMAdapter so API-driven flows work correctly.
    """

    supports_criteria: bool = True
    speech_passthrough: bool = False

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        system_prompt: str = "",
        environment_variables: dict[str, str] | None = None,
    ) -> None:
        self._model_id = model_id
        self._system_prompt = system_prompt
        self.responses: list[str] = []
        self.session_ended: bool = False
        self._machine: Any = None  # Set by DialogMachine after from_flow()
        # HTTP action execution state (same as LLMAdapter)
        self._env_vars: dict[str, str] = dict(environment_variables or {})
        self._jinja_env = Environment(loader=BaseLoader(), undefined=ChainableUndefined)
        # GET-only URL cache: keyed by "METHOD:rendered_url".
        # Prevents duplicate GET calls with identical parameters (e.g. courses-by-city
        # firing twice when chain visits list_courses_in_city then ask_course_preference).
        # Cache key includes the full URL so a city change (different URL) bypasses cache.
        self._get_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Template helpers (mirrors LLMAdapter)
    # ------------------------------------------------------------------

    def _render(self, template_str: str, context: dict[str, Any]) -> str:
        try:
            return self._jinja_env.from_string(template_str).render(**context)
        except Exception as exc:
            logger.warning("ToolCallAdapter: template render failed for %r: %s", template_str[:60], exc)
            return template_str

    def _build_context(self, userdata: dict[str, Any]) -> dict[str, Any]:
        """Merge env_vars + userdata into a flat Jinja2 context."""
        ctx: dict[str, Any] = {}
        ctx.update(self._env_vars)
        ctx.update(userdata)
        return ctx

    # ------------------------------------------------------------------
    # Instruction builder
    # ------------------------------------------------------------------

    def _build_instructions(self, node: FlowNode, machine: Any) -> str:
        """Build instructions identical to SimpleFlowAgent.__init__."""
        lang = _resolve_lang(machine)

        flow_system_prompt = getattr(machine._flow, "system_prompt", "") or ""
        flow_system_prompt = _process_text(flow_system_prompt, machine, lang)

        if node.static_text:
            node_instruction = (
                "A message has been spoken to the user. "
                "Wait for their response, then use the appropriate tool "
                "to transition to the next step."
            )
        elif node.instruction:
            node_instruction = _process_text(node.instruction, machine, lang)
        else:
            node_instruction = ""

        if node.edges and not node.is_final:
            edge_lines = [f'  - "{e.id}": {e.condition}' for e in node.edges]
            node_instruction += "\n\nAvailable transitions:\n" + "\n".join(edge_lines)

        # Inject current date/time so LLM resolves partial dates (e.g. "28 May") correctly.
        time_ctx = _get_time_context()
        time_line = (
            f"[TODAY] {time_ctx['current_date']}  |  "
            f"IST {time_ctx['current_time_Asia_Kolkata']}"
        )

        base = f"{flow_system_prompt}\n\n{node_instruction}".strip() if flow_system_prompt else node_instruction
        return f"{time_line}\n\n{base}"

    # ------------------------------------------------------------------
    # Adapter surface
    # ------------------------------------------------------------------

    async def speak(self, text: str, node: FlowNode) -> None:
        """Record static text as a response."""
        self.responses.append(text)

    async def generate_reply(
        self,
        instruction: str,
        node: FlowNode,
        history: list[dict] | None = None,
        userdata: dict | None = None,
    ) -> str:
        """Generate speech for a node entry.

        Priority:
        1. [EN]/[HI] language markers → extract relevant language line (0 LLM calls)
        2. "Agent says: <text>" prefix → extract speech line (0 LLM calls)
        3. Call LLM with rendered instruction (1 LLM call — complex/template flows)
        """
        lang = _resolve_lang(self._machine) if self._machine else "en"

        # Render Jinja2 templates with current userdata before extraction/LLM
        rendered_instruction = instruction
        if userdata and "{{" in instruction:
            ctx = self._build_context(userdata)
            rendered_instruction = self._render(instruction, ctx)

        # Code-level zero-speech gate: if the rendered instruction explicitly
        # mandates silence (e.g. "ZERO speech — no words before the tool call")
        # AND all current_* template variables have non-empty values (meaning
        # the STEP 4 / auto-proceed condition is satisfied), skip the LLM entirely.
        # Relying on the LLM to output an empty string is unreliable.
        if _is_zero_speech_step(rendered_instruction):
            logger.debug("[ToolCallAdapter] zero-speech gate triggered — skipping entry speech")
            return ""

        # Try language markers first ([EN]/[HI])
        # Skip shortcut when multiple blocks for the same language exist —
        # multi-step nodes (e.g. category_healing_village with 4 [EN] blocks)
        # require LLM to pick the right step based on conversation progress.
        _lang_match_count = sum(
            1 for m in _LANG_MARKER_RE.finditer(rendered_instruction)
            if m.group(1).lower() == lang or (lang != "en" and m.group(1).lower() == "en")
        )
        extracted = _extract_speech_text(rendered_instruction, self._machine, lang)
        if extracted and _lang_match_count <= 2:  # allow [EN]+[HI] pair — that's 1 step
            self.responses.append(extracted)
            return extracted

        # Try "Agent says: <text>" prefix (BOB-style flows)
        agent_says = _extract_agent_says(rendered_instruction)
        if agent_says:
            self.responses.append(agent_says)
            return agent_says

        # Complex instruction — call LLM to generate natural reply
        return await self._generate_via_llm(rendered_instruction, history or [])

    async def _generate_via_llm(self, instruction: str, history: list[dict]) -> str:
        """Call LLM to generate entry speech for complex/template instructions."""
        from openai import AsyncOpenAI

        speech_directive = (
            "SPEECH GENERATION MODE: Generate ONLY the agent's natural spoken response. "
            "Do NOT output tool call syntax, JSON objects, function names, routing "
            "decisions, or edge IDs. Output only what the agent would SAY to the caller. "
            "CRITICAL SILENCE RULE: If the instruction contains 'ZERO speech', "
            "'zero words', 'no words before the tool call', 'call the tool immediately', "
            "'Do NOT speak', 'Silent routing node', or any directive to produce NO speech "
            "— output an EMPTY string. Never ask for information when instructed to be silent."
        )
        base = f"{self._system_prompt}\n\n{instruction}" if self._system_prompt else instruction
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": f"{speech_directive}\n\n{base}",
            }
        ]
        messages.extend(history[-6:])

        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        _t0 = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=_strip_provider_prefix(self._model_id),
                messages=messages,
                temperature=0.3,
            )
        except Exception as exc:
            logger.error("[ToolCallAdapter] generate_reply LLM call failed: %s", exc)
            return instruction

        latency_ms = (time.perf_counter() - _t0) * 1000
        usage = getattr(response, "usage", None)
        print(
            f"[LLM] {latency_ms:.0f}ms  "
            f"in={getattr(usage, 'prompt_tokens', 0)} "
            f"out={getattr(usage, 'completion_tokens', 0)} tok  "
            f"model={self._model_id}"
        )
        text = response.choices[0].message.content or ""
        self.responses.append(text)
        return text

    async def evaluate_criteria(
        self,
        node: FlowNode,
        history: list[dict[str, Any]],
        userdata: dict[str, Any],
    ) -> CriteriaResult:
        """Evaluate via LLM tool-calling (mirrors SimpleFlowAgent)."""
        from openai import AsyncOpenAI

        machine = self._machine

        # Build tool schemas from descriptors
        if machine:
            descriptors = machine.get_tools_for_node(node)
        else:
            descriptors = [
                ToolDescriptor(
                    id=e.id,
                    description=e.condition,
                    is_data_collection=e.input_schema is not None,
                    input_schema=(
                        e.input_schema if isinstance(e.input_schema, dict) else None
                    ),
                    target_node_id=e.target_node_id,
                )
                for e in node.edges
            ]

        tools = _descriptors_to_openai_tools(descriptors)
        if not tools:
            return CriteriaResult(node_id=node.id)

        instructions = (
            self._build_instructions(node, machine) if machine else self._system_prompt
        )

        # Inject non-meta userdata as explicit context so router nodes
        # can reliably check slot values (e.g. {{name}}) without relying
        # solely on Jinja2 template rendering, which can confuse the LLM.
        _slot_ctx = {k: v for k, v in userdata.items() if k != "_flow_meta" and v not in (None, "")}
        if _slot_ctx:
            _slot_lines = "\n".join(f"  {k}: {v}" for k, v in _slot_ctx.items())
            instructions = f"{instructions}\n\n[CURRENT DATA]\n{_slot_lines}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": instructions},
        ]
        for msg in history[-10:]:
            messages.append(msg)

        client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        _t0 = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=_strip_provider_prefix(self._model_id),
                messages=messages,
                tools=tools,
                tool_choice="required",
                temperature=0,
            )
        except Exception as exc:
            logger.error("[ToolCallAdapter] LLM call failed: %s", exc)
            return CriteriaResult(node_id=node.id)

        latency_ms = (time.perf_counter() - _t0) * 1000
        usage = getattr(response, "usage", None)
        print(
            f"[LLM] {latency_ms:.0f}ms  "
            f"in={getattr(usage, 'prompt_tokens', 0)} "
            f"out={getattr(usage, 'completion_tokens', 0)} tok  "
            f"model={self._model_id}"
        )

        choice = response.choices[0]

        if not choice.message.tool_calls:
            logger.info("[ToolCallAdapter] no tool_call returned for node=%s", node.id)
            return CriteriaResult(node_id=node.id)

        tool_call = choice.message.tool_calls[0]
        edge_id = tool_call.function.name
        extracted_slots: dict[str, Any] = {}

        if tool_call.function.arguments:
            try:
                extracted_slots = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                pass

        # Strip keys not in the edge's schema — prevents LLM hallucinating extra
        # fields (e.g. collected_name: "unknown") that pollute userdata and cause
        # downstream nodes to appear pre-filled, skipping required user interactions.
        matched_descriptor = next(
            (d for d in descriptors if d.id == edge_id), None
        )
        if matched_descriptor and matched_descriptor.input_schema:
            allowed_keys = set(
                matched_descriptor.input_schema.get("properties", {}).keys()
            )
            if allowed_keys:
                extracted_slots = {
                    k: v for k, v in extracted_slots.items() if k in allowed_keys
                }

        logger.info(
            "[ToolCallAdapter] tool_call=%s slots=%s node=%s",
            edge_id,
            extracted_slots,
            node.id,
        )

        return CriteriaResult(
            node_id=node.id,
            recommended_edge_id=edge_id,
            all_required_met=True,
            extracted_slots=extracted_slots,
        )

    async def execute_action(
        self,
        action: CustomAction,
        userdata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Execute an HTTP action — identical logic to LLMAdapter.execute_action."""
        import re as _re

        import httpx

        print(f"[TRACK] ToolCallAdapter.execute_action START - action_id: {action.id}, userdata keys: {list(userdata.keys()) if userdata else 'None'}")

        # run_once: return cached result if already succeeded this session
        if action.run_once and action.store_response_as:
            cached = userdata.get(action.store_response_as, {})
            if isinstance(cached, dict) and cached.get("success"):
                print(f"[TRACK] ToolCallAdapter - run_once HIT for action={action.id}, returning cached result")
                return cached

        ctx = self._build_context(userdata)

        # condition guard
        if action.condition:
            condition_result = self._render(action.condition, ctx)
            if not condition_result.strip():
                print(f"[TRACK] ToolCallAdapter - action={action.id} SKIPPED (condition empty after render)")
                return None

        url = self._render(action.url, ctx)
        headers = {k: self._render(v, ctx) for k, v in action.headers.items()}
        method = action.method.value if hasattr(action.method, "value") else str(action.method)

        print(f"[TRACK] ToolCallAdapter - rendered URL: {url}  (template: {action.url[:80]})")
        print(f"[TRACK] ToolCallAdapter - method: {method}")

        # GET cache: same URL in same session → return cached successful result.
        # Key includes the full rendered URL so a city/param change produces a
        # different key and bypasses the cache (fires the API again as needed).
        if method.upper() == "GET":
            _cache_key = f"GET:{url}"
            if _cache_key in self._get_cache:
                print(f"[TRACK] ToolCallAdapter - GET cache HIT for action={action.id} url={url}")
                return self._get_cache[_cache_key]

        body: Any = None
        if action.body_template:
            body_str = self._render(action.body_template, ctx)
            print(f"[TRACK] ToolCallAdapter - rendered body: {body_str[:200]}")
            try:
                body = json.loads(body_str)
                if isinstance(body, dict):
                    body = _coerce_numeric_strings(body, set(action.string_fields))
            except json.JSONDecodeError:
                body = body_str

        try:
            async with httpx.AsyncClient(timeout=float(action.timeout)) as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, dict) else None,
                    content=body.encode() if isinstance(body, str) else None,
                )
                result: dict[str, Any] = {
                    "status": response.status_code,
                    "success": response.status_code < 400,
                    "headers": dict(response.headers),
                    "_rendered_url": url,
                    "_method": method,
                }
                try:
                    result["data"] = response.json()
                except Exception:
                    result["data"] = response.text

                status_tag = "OK" if response.status_code < 400 else "FAILED"
                print(f"[TRACK] ToolCallAdapter - action={action.id} {status_tag} status={response.status_code}")
                print(f"[TRACK] ToolCallAdapter - response data: {json.dumps(result.get('data'), default=str)[:500]}")

                # Apply env_updates (e.g. store ACCESS_TOKEN for subsequent actions)
                for update in action.env_updates:
                    value: Any = result
                    try:
                        for key in update.result_path.split("."):
                            value = value[key]
                        self._env_vars[update.env_key] = str(value)
                        print(f"[TRACK] ToolCallAdapter - env_update: {update.env_key} = {str(value)[:80]}")
                    except (KeyError, TypeError, IndexError):
                        print(f"[TRACK] ToolCallAdapter - env_update FAILED: could not resolve {update.result_path} for {update.env_key}")

                # Populate GET cache for successful responses.
                # Key is "GET:<rendered_url>" — city change → different URL → different key.
                if method.upper() == "GET" and result["success"]:
                    self._get_cache[f"GET:{url}"] = result

                return result

        except Exception as exc:
            logger.error("[ToolCallAdapter] action=%s HTTP error: %s", action.id, exc)
            return {
                "success": False,
                "error": str(exc),
                "status": 0,
                "_rendered_url": url,
                "_method": method,
            }

    async def end_session(self) -> None:
        """Mark session as ended."""
        self.session_ended = True
