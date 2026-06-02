"""LLM provider layer for superdialog."""

from .litellm_provider import LitellmProvider
from .provider import CompletionResult, LLMProvider, StreamChunk
from .registry import CustomProviderConfig, get_custom, register_llm_provider
from .resolver import resolve_llm

__all__ = [
    "CompletionResult",
    "CustomProviderConfig",
    "LLMProvider",
    "LitellmProvider",
    "StreamChunk",
    "get_custom",
    "register_llm_provider",
    "resolve_llm",
]
