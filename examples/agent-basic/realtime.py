"""Run a basic tool-calling agent with DoublewordLLM (real-time).

A simple agent — LLM + calculator tool — invoked through Doubleword's
real-time `/v1/chat/completions` endpoint. Runs a single query and a
sequential batch of five, printing wall time for each.

Requires DOUBLEWORD_API_KEY in the environment (or ~/.dw/credentials.toml).
"""

import asyncio
import os
import time

from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

from llamaindex_doubleword import DoublewordLLM

MODEL = "Qwen/Qwen3-14B-FP8"

QUERIES = [
    "What is 137 * 49?",
    "What is 100 + 250?",
    "What is 81 / 9?",
    "What is 2 ** 10?",
    "What is 1000 - 333?",
]


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Supports +, -, *, /, **, parentheses, and integer/float literals.
    """
    if not all(c in "0123456789+-*/.()** " for c in expression):
        return f"error: invalid characters in {expression!r}"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as e:
        return f"error: {e}"


async def main() -> None:
    if not os.environ.get("DOUBLEWORD_API_KEY") and not (
        os.path.exists(os.path.expanduser("~/.dw/credentials.toml"))
        and os.path.exists(os.path.expanduser("~/.dw/config.toml"))
    ):
        raise SystemExit(
            "DOUBLEWORD_API_KEY not set and no ~/.dw/credentials.toml found."
        )

    llm = DoublewordLLM(model=MODEL, temperature=0, is_function_calling_model=True)
    calc_tool = FunctionTool.from_defaults(fn=calculator)
    agent = AgentWorkflow.from_tools_or_functions(
        [calc_tool],
        llm=llm,
    )

    print("=" * 60)
    print("DoublewordLLM (real-time)")
    print(f"Model: {MODEL}")
    print("=" * 60)
    print()

    # Single query
    print("--- single query ---")
    start = time.monotonic()
    response = await agent.run(QUERIES[0])
    elapsed = time.monotonic() - start
    answer = str(response)
    print(f"  wall time: {elapsed:5.1f}s")
    print(f"  Q: {QUERIES[0]}")
    print(f"  A: {answer[:120]}")
    print()

    # Sequential queries
    print(f"--- {len(QUERIES)} queries (sequential) ---")
    start = time.monotonic()
    answers = []
    for q in QUERIES:
        resp = await agent.run(q)
        answers.append(str(resp))
    elapsed = time.monotonic() - start
    print(f"  wall time: {elapsed:5.1f}s")
    for q, a in zip(QUERIES, answers):
        print(f"  Q: {q}")
        print(f"  A: {a[:120]}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
