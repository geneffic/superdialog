"""Machine adapters -- runtime backends for ``DialogStateMachine``."""

from .llm_adapter import LLMAdapter
from .text_adapter import TextAdapter

__all__ = ["LLMAdapter", "TextAdapter"]
