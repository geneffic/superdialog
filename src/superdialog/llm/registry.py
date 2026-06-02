"""Custom-provider registry for resolver."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CustomProviderConfig:
    base_url: str
    api_key: str
    api_style: str = "openai"


_REGISTRY: dict[str, CustomProviderConfig] = {}


def register_llm_provider(
    name: str, base_url: str, api_key: str, api_style: str = "openai"
) -> None:
    _REGISTRY[name] = CustomProviderConfig(
        base_url=base_url, api_key=api_key, api_style=api_style
    )


def get_custom(name: str) -> CustomProviderConfig | None:
    return _REGISTRY.get(name)
