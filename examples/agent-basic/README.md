# agent-basic

A minimal tool-calling agent example. Two scripts run the same agent
(LLM + calculator tool) against the two `llamaindex-doubleword` LLM classes,
printing wall time for each.

| Script        | Model                | Path                    | Concurrency      |
|---------------|----------------------|-------------------------|------------------|
| `realtime.py` | `DoublewordLLM`      | `/v1/chat/completions`  | sync, sequential |
| `batched.py`  | `DoublewordLLMBatch` | Doubleword batch API via `autobatcher` | async, `asyncio.gather` |

Each script runs:

1. **A single query** — `DoublewordLLM` is faster here because the
   `batch_window_seconds` of `DoublewordLLMBatch` is overhead with nothing
   to collate.
2. **Five queries** — sequential for `realtime.py`, concurrent via
   `asyncio.gather` for `batched.py`. The five concurrent calls to
   `DoublewordLLMBatch` get collated into a single autobatcher window and
   submitted as one batch.

## Running

```bash
export DOUBLEWORD_API_KEY="sk-..."   # or use ~/.dw/credentials.toml

cd examples/agent-basic
uv sync
uv run python realtime.py
uv run python batched.py
```

Edit `MODEL` at the top of either script to point at whichever model you
have access to.
