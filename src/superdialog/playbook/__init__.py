"""Playbook engine: declarative journeys behind the public Agent protocol."""

from .agent import PlaybookAgent
from .compiler import compile_flow, coverage_report
from .director import CompletesLLM
from .eval_bridge import (
    EvalReport,
    PersonaSpec,
    SessionMetrics,
    run_eval,
    run_session,
)
from .events import EventLog
from .models import Playbook
from .providers import ProviderDirector, ProviderTalker, provider_adapters
from .replay import ReplayReport, replay
from .simple import is_simple_playbook, load_simple, simple_to_playbook
from .state import ConversationState
from .talker import StreamsLLM
from .toolexec import HttpFn, PythonToolFn, httpx_http

__all__ = [
    "CompletesLLM",
    "ConversationState",
    "EvalReport",
    "EventLog",
    "HttpFn",
    "PersonaSpec",
    "Playbook",
    "PlaybookAgent",
    "ProviderDirector",
    "ProviderTalker",
    "PythonToolFn",
    "ReplayReport",
    "SessionMetrics",
    "StreamsLLM",
    "compile_flow",
    "coverage_report",
    "httpx_http",
    "is_simple_playbook",
    "load_simple",
    "provider_adapters",
    "replay",
    "run_eval",
    "run_session",
    "simple_to_playbook",
]
