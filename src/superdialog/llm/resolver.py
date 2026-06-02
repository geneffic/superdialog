"""Resolve a model URI into an LLMProvider instance."""

from __future__ import annotations

from .litellm_provider import LitellmProvider
from .provider import LLMProvider
from .registry import get_custom


def resolve_llm(uri: str) -> LLMProvider:
    """Parse a model URI and return an LLMProvider instance.

    Examples:
        openai/gpt-5.1                          -> LitellmProvider("openai/gpt-5.1")
        anthropic/claude-opus-4-7               -> LitellmProvider("anthropic/...")
        custom/<name>/<model>                   -> uses registered base_url + api_key
        vllm/<model>@<host>                     -> hosted_vllm via api_base
        ollama/<model>@<host>                   -> ollama via api_base
    """
    if uri.startswith("custom/"):
        parts = uri.split("/", 2)
        if len(parts) < 3:
            raise ValueError(f"Custom URI requires model: {uri}")
        _, name, model = parts
        cfg = get_custom(name)
        if not cfg:
            raise ValueError(f"Unknown custom provider: {name}")
        return LitellmProvider(
            model=f"openai/{model}", api_base=cfg.base_url, api_key=cfg.api_key
        )
    if "@" in uri:
        provider_model, host = uri.split("@", 1)
        scheme, model = provider_model.split("/", 1)
        litellm_scheme = {"vllm": "hosted_vllm", "ollama": "ollama"}.get(scheme, scheme)
        return LitellmProvider(model=f"{litellm_scheme}/{model}", api_base=host)
    return LitellmProvider(model=uri)
