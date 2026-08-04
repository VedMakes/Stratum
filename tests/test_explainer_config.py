import pytest

from stratum import explainer


def test_llm_provider_requires_configuration(monkeypatch):
    for name in (
        "STRATUM_LLM_PROVIDER", "STRATUM_LLM_API_KEY", "STRATUM_LLM_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "OPENAI_API_KEY", "OPENAI_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="STRATUM_LLM_PROVIDER"):
        explainer._provider()


def test_generic_llm_config_takes_precedence(monkeypatch):
    monkeypatch.setenv("STRATUM_LLM_PROVIDER", "openai")
    monkeypatch.setenv("STRATUM_LLM_API_KEY", "generic-key")
    monkeypatch.setenv("STRATUM_LLM_MODEL", "generic-model")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    monkeypatch.setenv("OPENAI_MODEL", "provider-model")

    provider = explainer._provider()

    assert provider == "openai"
    assert explainer._api_key(provider) == "generic-key"
    assert explainer._model(provider) == "generic-model"


def test_provider_specific_fallbacks(monkeypatch):
    monkeypatch.delenv("STRATUM_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("STRATUM_LLM_API_KEY", raising=False)
    monkeypatch.delenv("STRATUM_LLM_MODEL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "legacy-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "legacy-model")

    provider = explainer._provider()

    assert provider == "anthropic"
    assert explainer._api_key(provider) == "legacy-key"
    assert explainer._model(provider) == "legacy-model"
