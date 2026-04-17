# Async Agents -- Research Orchestrator

A LlamaIndex + `llamaindex-doubleword` adaptation of the
[Doubleword async-agents workbook](https://docs.doubleword.ai/inference-api/async-agents).

A single root agent breaks a research topic into sub-queries, spawns parallel
sub-agents (each of which can recursively spawn its own sub-agents up to
`MAX_DEPTH=3`), and synthesises a final markdown report. All concurrent LLM
calls are transparently batched through Doubleword's batch API via
`DoublewordLLMBatch` and `autobatcher`.

## How it works

1. The topic is pre-searched via the Serper API before any agent is created.
2. The root agent reviews search results and decides which angles to research.
3. It calls `spawn_agents` with a list of sub-queries -- each sub-agent is
   created with its own pre-searched results and runs in parallel via
   `asyncio.gather`.
4. Sub-agents can read pages (Jina Reader), search for new angles, spawn their
   own sub-agents, or reference another agent's completed findings.
5. When all sub-agents complete, the root synthesises a markdown report.
6. Output: `results/` directory with `report.md`, `agent-tree.json`, and
   `summary.json`.

## Prerequisites

```bash
export DOUBLEWORD_API_KEY="sk-..."   # or use ~/.dw/credentials.toml
export SERPER_API_KEY="..."           # https://serper.dev
```

## Running

```bash
cd examples/async-agents-research
uv sync
uv run python research.py
```

Edit `TOPIC` and `MODEL` at the top of `research.py` to change the research
subject and model.

## Layout

```
async-agents-research/
├── pyproject.toml
├── README.md            <- you are here
├── research.py          <- main orchestrator script
├── prompts.py           <- system prompts for root and sub-agents
└── tools.py             <- Serper search + Jina Reader wrappers
```
