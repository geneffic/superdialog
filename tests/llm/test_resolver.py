import pytest

from superdialog.llm.litellm_provider import LitellmProvider
from superdialog.llm.registry import register_llm_provider
from superdialog.llm.resolver import resolve_llm


def test_openai_uri() -> None:
    p = resolve_llm("openai/gpt-5.1")
    assert isinstance(p, LitellmProvider)
    assert p.model == "openai/gpt-5.1"


def test_anthropic_uri() -> None:
    p = resolve_llm("anthropic/claude-opus-4-7")
    assert isinstance(p, LitellmProvider)
    assert p.model == "anthropic/claude-opus-4-7"


def test_vllm_with_host() -> None:
    p = resolve_llm("vllm/llama-3@http://my-vllm:8000")
    assert isinstance(p, LitellmProvider)
    assert p.model == "hosted_vllm/llama-3"
    assert p.default_opts.get("api_base") == "http://my-vllm:8000"


def test_ollama_with_host() -> None:
    p = resolve_llm("ollama/llama3@http://localhost:11434")
    assert isinstance(p, LitellmProvider)
    assert p.model == "ollama/llama3"
    assert p.default_opts.get("api_base") == "http://localhost:11434"


def test_custom_provider_requires_registration() -> None:
    with pytest.raises(ValueError, match="Unknown custom provider"):
        resolve_llm("custom/unknown/model")


def test_custom_provider_after_registration() -> None:
    register_llm_provider("kerali", "https://llm.kerali.io/v1", "key-123")
    p = resolve_llm("custom/kerali/llama-3-70b")
    assert isinstance(p, LitellmProvider)
    assert p.model == "openai/llama-3-70b"
    assert p.default_opts.get("api_base") == "https://llm.kerali.io/v1"
    assert p.default_opts.get("api_key") == "key-123"
