"""Large-language-model client adapters.

Implements :class:`~app.domain.ports.LLMClient` with two concrete backends:

* :class:`OpenAICompatibleClient` — talks to any OpenAI-compatible
  ``/chat/completions`` endpoint. This single client covers the brief's
  preferred and fallback models: IBM Granite and Llama 3 are both commonly
  served behind an OpenAI-compatible gateway (watsonx proxy, Ollama, vLLM,
  TGI). Streaming is supported via server-sent events.
* :class:`EchoClient` — a deterministic offline client that composes an answer
  strictly from the supplied context. It guarantees the pipeline (and its
  citation guarantees) works with zero external dependencies, and is the
  default in tests and air-gapped demos.

``build_llm_client`` maps the configured provider to a client and falls back
to :class:`EchoClient` when no endpoint/key is configured, so the app never
crashes for lack of a model.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.infrastructure import offline_writer
from app.shared.config import LLMProvider
from app.shared.exceptions import LLMError
from app.shared.logging import get_logger

logger = get_logger("llm")


class EchoClient:
    """Offline, deterministic client that grounds answers in the given context.

    It never invents facts. It inspects the task instruction and produces
    task-appropriate, grounded output (summaries, quizzes, flashcards,
    comparisons, extractions) via :mod:`app.infrastructure.offline_writer`,
    preserving the ``[n]`` citation markers so answers remain verifiable. When
    the context is empty it returns a clear "not found" message, which the RAG
    engine detects.
    """

    def __init__(self, model: str = "echo-offline") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return f"echo/{self._model}"

    def _compose(self, prompt: str) -> str:
        return offline_writer.compose(prompt)

    def complete(
        self, *, system: str, prompt: str  # noqa: ARG002
    ) -> tuple[str, int | None, int | None]:
        text = self._compose(prompt)
        return text, None, None

    def stream(self, *, system: str, prompt: str) -> Iterator[str]:  # noqa: ARG002
        for token in self._compose(prompt).split(" "):
            yield token + " "


class FallbackLLMClient:
    """Wraps a primary client and degrades to a fallback on failure.

    If the primary backend (e.g. a remote Granite/Llama endpoint) is
    unreachable or errors, requests transparently fall back to the offline
    :class:`EchoClient` so the user still receives a grounded, cited answer
    rather than an error. The degradation is logged, never silent.
    """

    def __init__(self, *, primary: OpenAICompatibleClient, fallback: EchoClient) -> None:
        self._primary = primary
        self._fallback = fallback

    @property
    def name(self) -> str:
        return self._primary.name

    def complete(self, *, system: str, prompt: str) -> tuple[str, int | None, int | None]:
        try:
            return self._primary.complete(system=system, prompt=prompt)
        except LLMError as exc:
            logger.warning(
                "LLM backend '{name}' unavailable ({err}); falling back to offline "
                "echo client. Start the endpoint or set LLM_PROVIDER=echo to silence.",
                name=self._primary.name,
                err=exc.message,
            )
            return self._fallback.complete(system=system, prompt=prompt)

    def stream(self, *, system: str, prompt: str) -> Iterator[str]:
        try:
            # Materialise eagerly so a mid-stream failure still triggers fallback.
            yield from list(self._primary.stream(system=system, prompt=prompt))
        except LLMError as exc:
            logger.warning(
                "LLM stream '{name}' unavailable ({err}); falling back to echo.",
                name=self._primary.name,
                err=exc.message,
            )
            yield from self._fallback.stream(system=system, prompt=prompt)


class OpenAICompatibleClient:
    """Client for any OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
    ) -> None:
        self._model = model
        self._url = api_base.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"openai_compatible/{self._model}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(self, system: str, prompt: str, *, stream: bool) -> dict[str, object]:
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
        }

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def complete(self, *, system: str, prompt: str) -> tuple[str, int | None, int | None]:
        try:
            response = httpx.post(
                self._url,
                headers=self._headers(),
                json=self._payload(system, prompt, stream=False),
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                "The language model returned an error.",
                details={"status": exc.response.status_code},
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                "Could not reach the language model backend.",
                details={"error": str(exc)},
            ) from exc

        try:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return (
                text,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("The language model returned an unexpected response shape.") from exc

    def stream(self, *, system: str, prompt: str) -> Iterator[str]:
        try:
            with httpx.stream(
                "POST",
                self._url,
                headers=self._headers(),
                json=self._payload(system, prompt, stream=True),
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[len("data:") :].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"]
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    content = delta.get("content")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise LLMError(
                "Streaming from the language model failed.",
                details={"error": str(exc)},
            ) from exc


# Provider -> default model routing. Granite is preferred, Llama is the
# fallback, both served through an OpenAI-compatible endpoint.
_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.GRANITE: "ibm-granite/granite-3.0-8b-instruct",
    LLMProvider.LLAMA: "meta-llama/Meta-Llama-3-8B-Instruct",
}


def build_llm_client(
    *,
    provider: LLMProvider,
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> EchoClient | FallbackLLMClient:
    """Construct the configured LLM client, degrading to Echo when offline."""
    if provider is LLMProvider.ECHO:
        return EchoClient(model=model)

    # Without a reachable endpoint we cannot call a real model; fall back so
    # the application still functions (grounded, citation-preserving answers).
    if not api_base:
        logger.warning(
            "No LLM endpoint configured for provider '{p}'; using offline Echo "
            "client. Set LLM_API_BASE for real generation.",
            p=provider.value,
        )
        return EchoClient(model=model)

    resolved_model = model or _DEFAULT_MODELS.get(provider, model)
    primary = OpenAICompatibleClient(
        model=resolved_model,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    # Wrap the real backend so an unreachable endpoint degrades to grounded,
    # cited offline answers instead of failing the request.
    return FallbackLLMClient(primary=primary, fallback=EchoClient(model=resolved_model))
