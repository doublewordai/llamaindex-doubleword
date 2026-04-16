# llamaindex-doubleword

A LlamaIndex integration package for [Doubleword](https://doubleword.ai).

This package wires Doubleword's OpenAI-compatible inference API
(`https://api.doubleword.ai/v1`) into LlamaIndex as both **real-time**
LLM / embedding models and **transparently-batched** variants powered by
[`autobatcher`](https://pypi.org/project/autobatcher/).

The batched variants are required to access models that Doubleword exposes
**only via the batch API**, and they cut cost on workloads that fan out
many concurrent calls — typically the case in agentic workflows.

## Installation

```bash
pip install llamaindex-doubleword
```

## Authentication

Three resolution paths, in precedence order:

1. **Explicit constructor argument**:
   ```python
   DoublewordLLM(model="...", api_key="sk-...")
   ```
2. **Environment variable**:
   ```bash
   export DOUBLEWORD_API_KEY=sk-...
   ```
3. **`~/.dw/credentials.toml`** — the same file written by Doubleword's CLI
   tooling. The active account is selected by `~/.dw/config.toml`'s
   `active_account` field, and `inference_key` from that account is used.

   ```toml
   # ~/.dw/config.toml
   active_account = "work"
   ```
   ```toml
   # ~/.dw/credentials.toml
   [accounts.work]
   inference_key = "sk-..."
   ```

   To use a non-active account from your credentials file, set
   `DOUBLEWORD_API_KEY` directly to that account's `inference_key` — there
   is no `account=` selector on the model itself.

## LLMs

### `DoublewordLLM` (real-time)

Drop-in LLM for any LlamaIndex workflow that expects an `LLM`.

```python
from llamaindex_doubleword import DoublewordLLM

llm = DoublewordLLM(model="your-model-name")

response = llm.complete("Explain bismuth in three sentences.")
print(response.text)
```

Tool calling is supported — use with LlamaIndex's agent framework:

```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool
from llamaindex_doubleword import DoublewordLLM

def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    return str(eval(expression, {"__builtins__": {}}, {}))

llm = DoublewordLLM(model="your-model-name")
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(fn=calculator)],
    llm=llm,
)

response = agent.run("What is 137 * 49?")
print(response)
```

### `DoublewordLLMBatch` (transparently batched)

Same interface, but every concurrent `.acomplete()` / `.achat()` call is
collected by `autobatcher` and submitted via Doubleword's batch endpoint.
**Async-only** — sync calls raise.

Use this when:

- The model you want is **batch-only** (some Doubleword-hosted models do not
  expose a real-time chat endpoint).
- You're running an agentic workflow with parallel branches and want
  ~50% cost savings via batch pricing.

```python
import asyncio
from llamaindex_doubleword import DoublewordLLMBatch

llm = DoublewordLLMBatch(model="batch-only-model")

async def main():
    # Concurrent calls collected into a single batch under the hood.
    results = await asyncio.gather(*[
        llm.acomplete(f"Summarize chapter {i}") for i in range(50)
    ])
    for r in results:
        print(r.text)

asyncio.run(main())
```

#### Tuning autobatcher

Four `autobatcher.BatchOpenAI` knobs are exposed as constructor arguments:

| Argument                | Default | Purpose                                                              |
|-------------------------|---------|----------------------------------------------------------------------|
| `batch_size`            | `1000`  | Submit a batch when this many requests are queued.                   |
| `batch_window_seconds`  | `10.0`  | Submit a batch after this many seconds even if the size cap is not reached. |
| `poll_interval_seconds` | `5.0`   | How often autobatcher polls for batch completion.                    |
| `completion_window`     | `"24h"` | Doubleword batch completion window. `"1h"` is more expensive but faster. |

```python
llm = DoublewordLLMBatch(
    model="your-model",
    batch_size=250,           # smaller batches for fast-turnaround nodes
    batch_window_seconds=2.5, # don't make latency-sensitive calls wait 10s
    completion_window="1h",   # pay more, finish quicker
)
```

The same arguments are available on `DoublewordEmbeddingBatch`.

## Embeddings

```python
from llamaindex_doubleword import DoublewordEmbedding, DoublewordEmbeddingBatch

embed = DoublewordEmbedding(model_name="your-embedding-model")
vec = embed.get_text_embedding("hello world")

# Or, transparently batched:
batch_embed = DoublewordEmbeddingBatch(model_name="your-embedding-model")
# vecs = await batch_embed.aget_text_embedding_batch([...])
```

## Use with LlamaIndex

`DoublewordLLM` and `DoublewordEmbedding` work with LlamaIndex's global
`Settings`:

```python
from llama_index.core import Settings, VectorStoreIndex

Settings.llm = DoublewordLLM(model="your-model")
Settings.embed_model = DoublewordEmbedding(model_name="your-embedding-model")

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is this about?")
```

## Configuration

| Argument    | Env var              | Default                          |
|-------------|----------------------|----------------------------------|
| `api_key`   | `DOUBLEWORD_API_KEY` | _required_                       |
| `api_base`  | `DOUBLEWORD_API_BASE`| `https://api.doubleword.ai/v1`   |
| `model`     | —                    | _required_                       |

All other arguments accepted by `llama_index.llms.openai_like.OpenAILike` are
forwarded unchanged (`temperature`, `max_tokens`, `timeout`, etc.).

## License

MIT
