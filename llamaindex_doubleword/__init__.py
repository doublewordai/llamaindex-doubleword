"""LlamaIndex integration for Doubleword.

This package wires Doubleword's OpenAI-compatible inference API into
LlamaIndex as both real-time LLM / embedding models and
transparently-batched variants powered by ``autobatcher``.
"""

from importlib import metadata

from llamaindex_doubleword.embeddings import (
    DoublewordEmbedding,
    DoublewordEmbeddingAsync,
    DoublewordEmbeddingBatch,
)
from llamaindex_doubleword.llm import (
    DEFAULT_DOUBLEWORD_API_BASE,
    DoublewordLLM,
    DoublewordLLMAsync,
    DoublewordLLMBatch,
)

try:
    __version__ = metadata.version(__package__ or "llamaindex-doubleword")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"
del metadata

__all__ = [
    "DEFAULT_DOUBLEWORD_API_BASE",
    "DoublewordEmbedding",
    "DoublewordEmbeddingAsync",
    "DoublewordEmbeddingBatch",
    "DoublewordLLM",
    "DoublewordLLMAsync",
    "DoublewordLLMBatch",
    "__version__",
]
