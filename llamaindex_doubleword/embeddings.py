"""Doubleword embedding model implementations for LlamaIndex.

Mirrors :mod:`llamaindex_doubleword.llm`: a real-time
:class:`DoublewordEmbedding` plus an async-only
:class:`DoublewordEmbeddingBatch` that routes through
:class:`autobatcher.BatchOpenAI` for transparent batching against
Doubleword's batch embeddings endpoint.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import Field, model_validator

from llamaindex_doubleword._credentials import resolve_api_key
from llamaindex_doubleword.llm import DEFAULT_DOUBLEWORD_API_BASE


def _resolve_api_key_str() -> str | None:
    """Resolve the API key as a plain string for LlamaIndex constructors."""
    resolved = resolve_api_key()
    if resolved is not None:
        return resolved.get_secret_value()
    return None


def _resolve_api_base() -> str:
    """Resolve the base URL from env or default."""
    return os.environ.get("DOUBLEWORD_API_BASE", DEFAULT_DOUBLEWORD_API_BASE)


class DoublewordEmbedding(OpenAIEmbedding):
    """Doubleword embedding model for LlamaIndex.

    A real-time embedding model targeting Doubleword's OpenAI-compatible API.
    Configure with ``DOUBLEWORD_API_KEY`` (or pass ``api_key=...``).

    Example:
        .. code-block:: python

            from llamaindex_doubleword import DoublewordEmbedding

            embed = DoublewordEmbedding(model_name="your-embedding-model")
            vectors = embed.get_text_embedding("Hello, world.")
    """

    def __init__(self, **kwargs: Any) -> None:
        # Apply Doubleword defaults before calling the parent constructor.
        if "api_key" not in kwargs:
            resolved = _resolve_api_key_str()
            if resolved is not None:
                kwargs["api_key"] = resolved
        if "api_base" not in kwargs:
            kwargs["api_base"] = _resolve_api_base()
        super().__init__(**kwargs)


class DoublewordEmbeddingBatch(DoublewordEmbedding):
    """Doubleword embedding model routed through the batch API via autobatcher.

    Concurrent ``aget_text_embedding`` / ``aget_text_embedding_batch`` calls
    are collected by :class:`autobatcher.BatchOpenAI` and submitted as a
    single batch to Doubleword.

    **Async-only.** Synchronous calls raise ``NotImplementedError``.
    Use the async variants from async code.

    Example:
        .. code-block:: python

            import asyncio
            from llamaindex_doubleword import DoublewordEmbeddingBatch

            embed = DoublewordEmbeddingBatch(model_name="your-embedding-model")

            async def main():
                vectors = await embed.aget_text_embedding_batch(
                    ["Hello", "World", "Foo", "Bar"]
                )
                print(len(vectors))

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
            "is not reached. Forwarded to autobatcher.BatchOpenAI."
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

    @model_validator(mode="after")
    def _install_autobatcher(self) -> DoublewordEmbeddingBatch:
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

        batch_client = BatchOpenAI(**client_kwargs)

        # LlamaIndex's OpenAI embedding stores the async client at _aclient
        # (a Pydantic PrivateAttr). Replace it with the BatchOpenAI instance
        # so async calls go through the batch pipeline.
        self._aclient = batch_client

        return self

    def get_text_embedding(self, text: str, **kwargs: Any) -> list[float]:
        raise NotImplementedError(
            "DoublewordEmbeddingBatch is async-only. Use `aget_text_embedding`."
        )

    def get_text_embedding_batch(
        self, texts: list[str], **kwargs: Any
    ) -> list[list[float]]:
        raise NotImplementedError(
            "DoublewordEmbeddingBatch is async-only. Use `aget_text_embedding_batch`."
        )

    def get_query_embedding(self, query: str, **kwargs: Any) -> list[float]:
        raise NotImplementedError(
            "DoublewordEmbeddingBatch is async-only. Use `aget_query_embedding`."
        )
