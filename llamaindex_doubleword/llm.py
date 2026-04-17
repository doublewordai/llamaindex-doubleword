"""Doubleword LLM implementations for LlamaIndex.

Two classes are exposed:

* :class:`DoublewordLLM` — a thin subclass of
  :class:`llama_index.llms.openai_like.OpenAILike` that targets Doubleword's
  OpenAI-compatible inference endpoint (``https://api.doubleword.ai/v1``) by
  default and reads ``DOUBLEWORD_API_KEY`` from the environment or
  ``~/.dw/credentials.toml``.

* :class:`DoublewordLLMBatch` — same surface, but the async client is replaced
  with :class:`autobatcher.BatchOpenAI`, which transparently collects concurrent
  requests and submits them through Doubleword's batch API. This is the only
  way to access models that Doubleword exposes solely via the batch endpoint,
  and it is the recommended choice for workflows that fan out many parallel
  calls.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from llama_index.llms.openai_like import OpenAILike
from pydantic import Field, SecretStr, model_validator

from llamaindex_doubleword._credentials import resolve_api_key

DEFAULT_DOUBLEWORD_API_BASE = "https://api.doubleword.ai/v1"


def _resolve_api_key_str() -> str | None:
    """Resolve the API key as a plain string for LlamaIndex constructors."""
    resolved = resolve_api_key()
    if resolved is not None:
        return resolved.get_secret_value()
    return None


def _resolve_api_base() -> str:
    """Resolve the base URL from env or default."""
    return os.environ.get("DOUBLEWORD_API_BASE", DEFAULT_DOUBLEWORD_API_BASE)


class DoublewordLLM(OpenAILike):
    """Doubleword LLM for LlamaIndex.

    A real-time chat/completion model targeting Doubleword's OpenAI-compatible
    API. Configure with ``DOUBLEWORD_API_KEY`` (or pass ``api_key=...``).

    Example:
        .. code-block:: python

            from llamaindex_doubleword import DoublewordLLM

            llm = DoublewordLLM(model="your-model")
            response = llm.complete("Hello, world.")
            print(response.text)
    """

    is_chat_model: bool = True
    is_function_calling_model: bool = True
    context_window: int = 128000

    def __init__(self, **kwargs: Any) -> None:
        # Apply Doubleword defaults before calling the parent constructor.
        if "api_key" not in kwargs:
            resolved = _resolve_api_key_str()
            if resolved is not None:
                kwargs["api_key"] = resolved
        if "api_base" not in kwargs:
            kwargs["api_base"] = _resolve_api_base()
        super().__init__(**kwargs)

    @property
    def metadata(self) -> Any:
        md = super().metadata
        return md


class DoublewordLLMBatch(DoublewordLLM):
    """Doubleword LLM that routes through the batch API via autobatcher.

    Concurrent ``acomplete`` / ``achat`` calls are collected by
    :class:`autobatcher.BatchOpenAI` and submitted as a single batch to
    Doubleword. This is the **only** way to use models that Doubleword exposes
    solely through batch endpoints, and the natural choice for workflows with
    parallel branches: collected calls amortize batch pricing (~50% cost
    savings) without changing user code.

    **Async-only.** Synchronous calls (``complete``, ``chat``, ``stream_complete``,
    ``stream_chat``) raise ``NotImplementedError`` because a 10-second batch
    collection window is incompatible with blocking call sites. Use
    ``acomplete`` / ``achat`` from async code instead.

    Example:
        .. code-block:: python

            import asyncio
            from llamaindex_doubleword import DoublewordLLMBatch

            llm = DoublewordLLMBatch(model="batch-only-model")

            async def main():
                results = await asyncio.gather(*[
                    llm.acomplete(f"Summarize chapter {i}") for i in range(50)
                ])
                for r in results:
                    print(r.text)

            asyncio.run(main())
    """

    batch_size: int = Field(
        default=1000,
        description=(
            "Submit a batch when this many requests have been queued. "
            "Forwarded to autobatcher.BatchOpenAI."
        ),
    )
    batch_window_seconds: float = Field(
        default=10.0,
        description=(
            "Submit a batch after this many seconds even if `batch_size` "
            "is not reached. Lower this to trade batch fullness for latency. "
            "Forwarded to autobatcher.BatchOpenAI."
        ),
    )
    poll_interval_seconds: float = Field(
        default=5.0,
        description=(
            "How often autobatcher polls Doubleword's batch endpoint for "
            "completion. Forwarded to autobatcher.BatchOpenAI."
        ),
    )
    completion_window: Literal["24h", "1h"] = Field(
        default="24h",
        description=(
            "Doubleword batch completion window. '1h' is more expensive "
            "but completes faster. Forwarded to autobatcher.BatchOpenAI."
        ),
    )
    _batch_client: Any = None

    @model_validator(mode="after")
    def _install_autobatcher(self) -> DoublewordLLMBatch:
        from autobatcher import BatchOpenAI

        api_key: str | None = self.api_key
        if api_key is None:
            api_key = _resolve_api_key_str()

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": self.api_base,
            "batch_size": self.batch_size,
            "batch_window_seconds": self.batch_window_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "completion_window": self.completion_window,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        if self.max_retries is not None and self.max_retries > 0:
            client_kwargs["max_retries"] = self.max_retries
        if self.additional_kwargs:
            if "default_headers" in self.additional_kwargs:
                client_kwargs["default_headers"] = self.additional_kwargs["default_headers"]

        self._batch_client = BatchOpenAI(**client_kwargs)

        return self

    def _get_aclient(self) -> Any:
        """Override to return the BatchOpenAI client instead of a regular AsyncOpenAI."""
        return self._batch_client

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DoublewordLLMBatch is async-only. Use `acomplete` from an async "
            "context. If you need a synchronous LLM, use `DoublewordLLM` instead."
        )

    def chat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DoublewordLLMBatch is async-only. Use `achat` from an async "
            "context. If you need a synchronous LLM, use `DoublewordLLM` instead."
        )

    def stream_complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DoublewordLLMBatch does not support streaming. Batch results "
            "return all at once when the batch completes. Use `DoublewordLLM` "
            "for streaming, or `acomplete` / `achat` for batched inference."
        )

    def stream_chat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DoublewordLLMBatch does not support streaming. Batch results "
            "return all at once when the batch completes. Use `DoublewordLLM` "
            "for streaming, or `acomplete` / `achat` for batched inference."
        )

    async def astream_complete(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DoublewordLLMBatch does not support streaming. Batch results "
            "return all at once when the batch completes. Use `DoublewordLLM` "
            "for streaming, or `acomplete` / `achat` for batched inference."
        )

    async def astream_chat(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "DoublewordLLMBatch does not support streaming. Batch results "
            "return all at once when the batch completes. Use `DoublewordLLM` "
            "for streaming, or `acomplete` / `achat` for batched inference."
        )
