from __future__ import annotations

import os


PROVIDER_ENV = "STRATUM_LLM_PROVIDER"
MODEL_ENV = "STRATUM_LLM_MODEL"
API_KEY_ENV = "STRATUM_LLM_API_KEY"
USER_LIMIT = 4000


def _anthropic_text(response) -> str:
    return "".join(getattr(block, "text", "") for block in response.content)


def _provider() -> str:
    provider = os.environ.get(PROVIDER_ENV)
    if provider:
        return provider.strip().lower()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        f"{PROVIDER_ENV} is not configured. Set {PROVIDER_ENV}, {MODEL_ENV}, "
        f"and {API_KEY_ENV}; provider-specific API key env vars are also supported."
    )


def _api_key(provider: str) -> str:
    key = os.environ.get(API_KEY_ENV)
    if key:
        return key
    provider_key = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider)
    if provider_key and os.environ.get(provider_key):
        return os.environ[provider_key]
    raise RuntimeError(f"{API_KEY_ENV} is not configured for provider '{provider}'")


def _model(provider: str) -> str:
    model = os.environ.get(MODEL_ENV)
    if model:
        return model
    provider_model = {
        "anthropic": "ANTHROPIC_MODEL",
        "openai": "OPENAI_MODEL",
    }.get(provider)
    if provider_model and os.environ.get(provider_model):
        return os.environ[provider_model]
    raise RuntimeError(f"{MODEL_ENV} is not configured for provider '{provider}'")


def _unsupported(provider: str) -> RuntimeError:
    return RuntimeError(
        f"Unsupported LLM provider '{provider}'. Supported providers: anthropic, openai."
    )


def call_llm(system: str, user: str, max_tokens: int = 600) -> str:
    provider = _provider()
    api_key = _api_key(provider)
    model = _model(provider)
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user[:USER_LIMIT]}],
        )
        return _anthropic_text(response)
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user[:USER_LIMIT]},
            ],
        )
        return response.choices[0].message.content or ""
    raise _unsupported(provider)


async def call_llm_async(system: str, user: str, max_tokens: int = 600) -> str:
    provider = _provider()
    api_key = _api_key(provider)
    model = _model(provider)
    if provider == "anthropic":
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user[:USER_LIMIT]}],
        )
        return _anthropic_text(response)
    if provider == "openai":
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user[:USER_LIMIT]},
            ],
        )
        return response.choices[0].message.content or ""
    raise _unsupported(provider)


def call_claude(system: str, user: str, max_tokens: int = 600) -> str:
    return call_llm(system, user, max_tokens=max_tokens)


async def call_claude_async(system: str, user: str, max_tokens: int = 600) -> str:
    return await call_llm_async(system, user, max_tokens=max_tokens)
