"""Execute ToolSpecs: template rendering, run_once/when policies, env_updates.

Tool templates render over {slots, env, results}. Unlike the Talker renderer,
env IS visible here: tools run Director-side and their output is never shown
to the Talker. Templates still come from playbook artifacts, so rendering is
sandboxed and template errors degrade to a failed ToolResultEvent.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from jinja2 import TemplateError, Undefined
from jinja2.sandbox import SandboxedEnvironment

from .events import EnvWriteEvent, Event, ToolCallEvent, ToolResultEvent
from .expr import ExprError, evaluate
from .models import SlotSpec, ToolSpec
from .state import ConversationState

HttpFn = Callable[..., Awaitable[tuple[int, Any]]]

# Sandboxed: tool templates are playbook artifacts (optimizer-generated), so
# attribute-walking SSTI payloads must be blocked, not executed.
_jinja = SandboxedEnvironment(undefined=Undefined, autoescape=False)

_CASTS: dict[str, Callable[[Any], Any]] = {
    "int": int,
    "float": float,
    "bool": lambda v: str(v).lower() in ("1", "true", "yes"),
    "str": str,
}


class PythonToolFn(Protocol):
    async def __call__(self, args: dict[str, Any], state: ConversationState) -> Any: ...


def _template_ns(state: ConversationState) -> dict[str, Any]:
    return {
        "slots": {k: v.value for k, v in state.slots.items()},
        "env": dict(state.env),
        "results": {
            k: {"ok": r.ok, "status": r.status, "data": r.data, "error": r.error}
            for k, r in state.tool_results.items()
        },
    }


def _render(template: str, ns: dict[str, Any]) -> str:
    return _jinja.from_string(template).render(**ns)


def _dig(data: Any, path: str) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def coerce_args(args: dict[str, Any], specs: dict[str, SlotSpec]) -> dict[str, Any]:
    """Cast incoming arg values to their declared SlotSpec types."""
    out = dict(args)
    for key, spec in specs.items():
        if key in out and spec.type in _CASTS:
            out[key] = _CASTS[spec.type](out[key])
    return out


class ToolExecutor:
    """Run a ToolSpec against state and return the events to append."""

    def __init__(
        self, http: HttpFn, python_tools: dict[str, PythonToolFn] | None = None
    ) -> None:
        self._http = http
        self._python_tools = python_tools or {}

    async def execute(
        self,
        spec: ToolSpec,
        state: ConversationState,
        args: dict[str, Any] | None = None,
    ) -> list[Event]:
        """Execute ``spec``; returns [] when run_once/when policies skip it."""
        if spec.run_once and state.tool_call_counts.get(spec.id, 0) > 0:
            return []
        if spec.when:
            try:
                if not evaluate(spec.when, state):
                    return []
            except ExprError:
                return []
        if args and spec.args:
            try:
                args = coerce_args(args, spec.args)
            except (TypeError, ValueError) as exc:
                return [
                    ToolCallEvent(tool=spec.id, args=args or {}),
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=False,
                        error=f"bad args: {exc}",
                    ),
                ]
        ns = _template_ns(state)
        events: list[Event] = []
        if spec.type == "python":
            fn = self._python_tools[spec.id]
            events.append(ToolCallEvent(tool=spec.id, args=args or {}))
            try:
                data = await fn(args or {}, state)
                events.append(
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=True,
                        data=data,
                    )
                )
            except Exception as exc:  # tool failure is data, not a crash
                events.append(
                    ToolResultEvent(
                        tool=spec.id,
                        store_as=spec.store_response_as,
                        ok=False,
                        error=str(exc),
                    )
                )
            return events

        try:
            url = _render(spec.url, ns)
            headers = {k: _render(v, ns) for k, v in spec.headers.items()}
            body = {
                k: _render(v, ns) if isinstance(v, str) else v
                for k, v in spec.body.items()
            } or None
        except TemplateError as exc:
            # Bad template (authoring typo or sandbox SecurityError) must not
            # crash the Director: record the attempt and a failed result.
            return [
                ToolCallEvent(tool=spec.id, args=args or {}),
                ToolResultEvent(
                    tool=spec.id,
                    store_as=spec.store_response_as,
                    ok=False,
                    error=f"template error: {exc}",
                ),
            ]
        events.append(
            ToolCallEvent(tool=spec.id, args={"url": url, "body": body or {}})
        )
        try:
            status, data = await self._http(
                method=spec.method,
                url=url,
                headers=headers,
                body=body,
                timeout=spec.timeout,
            )
        except Exception as exc:
            events.append(
                ToolResultEvent(
                    tool=spec.id,
                    store_as=spec.store_response_as,
                    ok=False,
                    error=str(exc),
                )
            )
            return events
        ok = 200 <= status < 300
        result = ToolResultEvent(
            tool=spec.id,
            store_as=spec.store_response_as,
            ok=ok,
            status=status,
            data=data,
            error=None if ok else str(data),
        )
        events.append(result)
        if ok:
            for env_key, path in spec.env_updates.items():
                value = _dig(data, path)
                if value is not None:
                    events.append(EnvWriteEvent(key=env_key, value=str(value)))
        return events


async def httpx_http(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: Any,
    timeout: float,
) -> tuple[int, Any]:
    """Production HTTP callable backed by httpx."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.request(method, url, headers=headers, json=body)
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"text": resp.text}
