"""Ollama-backed LLM interface for the rewriter pipeline.

Talks to the Ollama HTTP API (``/api/chat``).  The rest of the application
depends only on the :class:`LLMBackend` protocol so the concrete backend
can be swapped for llama.cpp, vLLM, or a mock in tests.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol

import httpx

from humanizer.config import Intensity, RewriterConfig
from humanizer.logging_config import get_logger

log = get_logger("llm_backend")


# ── Abstract protocol ────────────────────────────────────────────


class LLMBackend(Protocol):
    """Minimal interface every LLM backend must satisfy."""

    def health_check(self) -> bool:
        """Return True if the backend is reachable and has a model loaded."""
        ...

    def generate(
        self,
        prompt: str,
        system: str = "",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        json_schema: dict | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a completion and return the text.

        ``json_schema`` asks for structured output when the backend supports
        it; ``history`` holds earlier turns (used for format repair)."""
        ...


# ── System prompts ───────────────────────────────────────────────

_SYSTEM_PROMPTS: dict[Intensity, str] = {
    Intensity.LIGHT: (
        "You are a careful copy-editor. Rewrite the user's text to fix "
        "obvious redundancy, wordiness, and awkward phrasing. "
        "Preserve the original meaning, facts, names, numbers, and tone. "
        "Make minimal changes — only where they clearly improve clarity. "
        "Return ONLY the rewritten text, no commentary."
    ),
    Intensity.MODERATE: (
        "You are a skilled editor. Rewrite the user's text to improve "
        "readability, flow, and vocabulary variety while keeping the same "
        "meaning, facts, names, numbers, negations, and logical relationships. "
        "You may restructure sentences and use synonyms when context makes "
        "them safe. Do not invent new information or remove important details. "
        "Return ONLY the rewritten text, no commentary."
    ),
    Intensity.HEAVY: (
        "You are an expert editor performing a thorough rewrite. "
        "Significantly improve clarity, variety, and naturalness. "
        "You may restructure sentences, combine or split them, and change "
        "between active and passive voice where appropriate. "
        "Preserve ALL original facts, names, numbers, negations, "
        "uncertainty markers, causal claims, and logical relationships. "
        "Do not invent information or drop important details. "
        "Return ONLY the rewritten text, no commentary."
    ),
}


def get_system_prompt(intensity: Intensity) -> str:
    """Return the system prompt for a given rewrite intensity."""
    return _SYSTEM_PROMPTS[intensity]


# ── Ollama implementation ────────────────────────────────────────


class OllamaBackend:
    """Concrete LLM backend that talks to a local Ollama instance.

    Parameters:
        config: Pipeline configuration (model name, URL, timeouts, …).
    """

    def __init__(self, config: RewriterConfig | None = None) -> None:
        self.config = config or RewriterConfig()
        self._client = httpx.Client(
            base_url=self.config.ollama_base_url,
            timeout=httpx.Timeout(
                connect=10.0,
                read=self.config.request_timeout,
                write=30.0,
                pool=10.0,
            ),
        )

    # ── Protocol methods ─────────────────────────────────────────

    def health_check(self) -> bool:
        """Ping Ollama and verify the configured model is available."""
        try:
            resp = self._client.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            # Ollama returns names like "qwen3.5:4b" — match the tag.
            model = self.config.model_name
            found = any(model in m for m in models)
            if not found:
                log.warning(
                    "Model %r not found in Ollama. Available: %s",
                    model,
                    models,
                )
            return found
        except (httpx.HTTPError, Exception) as exc:
            log.error("Ollama health-check failed: %s", exc)
            return False

    def generate(
        self,
        prompt: str,
        system: str = "",
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        json_schema: dict | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Send a chat request to Ollama and return the assistant response.

        On failure, retries up to ``config.max_retries`` times, then
        raises :class:`LLMError`.
        """
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        seed = seed if seed is not None else self.config.seed

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "keep_alive": self.config.keep_alive,
            "think": self.config.llm_think,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                # Fix the context size so Ollama does not allocate a larger
                # default KV cache than an 8 GB machine can afford.
                "num_ctx": self.config.context_window,
            },
        }
        if self.config.num_threads:
            payload["options"]["num_thread"] = self.config.num_threads
        if json_schema is not None:
            payload["format"] = json_schema
        if seed is not None:
            payload["options"]["seed"] = seed

        last_exc: Exception | None = None
        attempts = 1 + self.config.max_retries

        for attempt in range(1, attempts + 1):
            try:
                t0 = time.monotonic()
                resp = self._client.post("/api/chat", json=payload)
                elapsed = time.monotonic() - t0
                resp.raise_for_status()

                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                if not content:
                    raise LLMError("Ollama returned an empty response.")

                log.debug(
                    "LLM generated %d chars in %.1fs (attempt %d)",
                    len(content),
                    elapsed,
                    attempt,
                )
                return content

            except httpx.TimeoutException as exc:
                last_exc = exc
                log.warning(
                    "Ollama timed out (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    exc,
                )
            except httpx.HTTPStatusError as exc:
                body = exc.response.text.lower() if exc.response is not None else ""
                if any(k in body for k in _MEMORY_MARKERS):
                    raise LLMMemoryError(
                        f"Ollama could not load {self.config.model_name!r}: not enough memory. "
                        "Try a smaller/quantized model or lower context_window."
                    ) from exc
                if exc.response is not None and exc.response.status_code == 404:
                    raise LLMError(
                        f"Model {self.config.model_name!r} is not installed. Run: ollama pull {self.config.model_name}"
                    ) from exc
                last_exc = exc
                log.warning(
                    "Ollama HTTP %d (attempt %d/%d): %s",
                    exc.response.status_code,
                    attempt,
                    attempts,
                    exc,
                )
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Ollama error (attempt %d/%d): %s",
                    attempt,
                    attempts,
                    exc,
                )

        raise LLMError(
            f"All {attempts} attempts failed. Last error: {last_exc}"
        ) from last_exc

    def unload(self) -> None:
        """Ask Ollama to free the model's memory now (keep_alive=0)."""
        try:
            self._client.post("/api/generate", json={"model": self.config.model_name, "keep_alive": 0})
        except httpx.HTTPError as exc:
            log.debug("Model unload failed: %s", exc)

    def close(self) -> None:
        """Close the HTTP client (and unload the model if configured)."""
        if self.config.unload_model_on_close:
            self.unload()
        self._client.close()


# ── Exceptions ───────────────────────────────────────────────────


class LLMError(RuntimeError):
    """Raised when the LLM backend fails irrecoverably."""


class LLMMemoryError(LLMError):
    """The model could not be loaded or run for lack of memory (not retried)."""


_MEMORY_MARKERS = ("out of memory", "requires more system memory", "insufficient memory", "cudamalloc")
