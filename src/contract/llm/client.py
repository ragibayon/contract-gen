from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

LLMProvider = Literal["openai", "openrouter", "ollama", "minsky", "anthropic"]


DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "openrouter": "openai/gpt-4o-mini",
    "ollama": "qwen2.5-coder:latest",
    "minsky": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "anthropic": "claude-haiku-4-5",
}


@dataclass
class LLMConfig:
    provider: LLMProvider = "ollama"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 30
    max_retries: int = 1


@dataclass
class LLMCallResult:
    text: str
    provider: str
    model: str
    elapsed_sec: float
    attempts: int
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_llm_config(
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> LLMConfig:
    selected_provider = provider or os.getenv("LLM_PROVIDER", "ollama")

    if selected_provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unsupported LLM provider: {selected_provider}. "
            f"Supported: {', '.join(DEFAULT_MODELS)}"
        )

    selected_model = (
        model
        or os.getenv(f"{selected_provider.upper()}_MODEL")
        or os.getenv("LLM_MODEL")
        or DEFAULT_MODELS[selected_provider]
    )

    return LLMConfig(
        provider=selected_provider,
        model=selected_model,
        temperature=_env_float("LLM_TEMPERATURE", 0.0),
        max_tokens=max_tokens or _env_int("LLM_MAX_OUTPUT_TOKENS", 2048),
        timeout=_env_int("LLM_TIMEOUT", 30),
        max_retries=_env_int("LLM_MAX_RETRIES", 1),
    )


class LLMClient:
    """
    Small bounded LLM wrapper.

    Design rules:
    - no unbounded loops
    - no provider-specific logic inside agents
    - raw text output by default
    - local parser/validator handles JSON later
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or load_llm_config()
        self.llm = self._build_model()

    def _build_model(self):
        provider = self.config.provider
        model = self.config.model or DEFAULT_MODELS[provider]

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=0,
            )

        if provider == "openrouter":
            from langchain_openai import ChatOpenAI

            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY is not set.")

            return ChatOpenAI(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=0,
                api_key=api_key,
                base_url=os.getenv(
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1",
                ),
                default_headers={
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
                    "X-OpenRouter-Title": os.getenv(
                        "OPENROUTER_SITE_NAME",
                        "contrAct",
                    ),
                },
            )

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=model,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens,
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            )

        if provider == "minsky":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=0,
                api_key=os.getenv("MINSKY_API_KEY", "EMPTY"),
                base_url=os.getenv("MINSKY_BASE_URL", "http://127.0.0.1:8001/v1"),
            )

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                max_retries=0,
            )

        raise ValueError(f"Unsupported provider: {provider}")

    def invoke(self, prompt: str, system: str | None = None) -> LLMCallResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))

        started = time.time()
        last_error: str | None = None

        max_attempts = max(1, self.config.max_retries + 1)

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.llm.invoke(messages)
                text = getattr(response, "content", str(response))

                return LLMCallResult(
                    text=str(text).strip(),
                    provider=self.config.provider,
                    model=self.config.model or DEFAULT_MODELS[self.config.provider],
                    elapsed_sec=round(time.time() - started, 3),
                    attempts=attempt,
                    ok=True,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        return LLMCallResult(
            text="",
            provider=self.config.provider,
            model=self.config.model or DEFAULT_MODELS[self.config.provider],
            elapsed_sec=round(time.time() - started, 3),
            attempts=max_attempts,
            ok=False,
            error=last_error,
        )


def ask_llm(
    prompt: str,
    system: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> LLMCallResult:
    config = load_llm_config(provider=provider, model=model, max_tokens=max_tokens)
    client = LLMClient(config)
    return client.invoke(prompt=prompt, system=system)
