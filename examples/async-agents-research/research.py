"""Recursive multi-agent research orchestrator.

Uses LlamaIndex AgentWorkflow + DoublewordLLMBatch to run a tree of research
agents that fan out via asyncio.gather. All concurrent LLM calls are
transparently batched through Doubleword's batch API via autobatcher.

Usage:
    uv run python research.py

Edit TOPIC and MODEL below to change the research subject and model.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.tools import FunctionTool

from llamaindex_doubleword import DoublewordLLMBatch

from prompts import ROOT_AGENT_SYSTEM, SUB_AGENT_SYSTEM
from tools import fetch_urls, format_results_for_context, search as serper_search

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "benefits of batch inference"
MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
MAX_DEPTH = 3
MAX_ITERATIONS = 8
OUTPUT_DIR = Path("results") / TOPIC.lower().replace(" ", "-")[:50]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

llm = DoublewordLLMBatch(
    model=MODEL,
    temperature=0,
    max_tokens=4096,
    is_function_calling_model=True,
    completion_window="1h",
    batch_window_seconds=2.5,
)

# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

SESSION_REGISTRY: dict = {}
SESSION_START: float = 0.0


def _elapsed() -> float:
    return time.monotonic() - SESSION_START


def _reset_session() -> None:
    global SESSION_START
    SESSION_REGISTRY.clear()
    SESSION_START = time.monotonic()


def _next_agent_id(prefix: str) -> str:
    n = SESSION_REGISTRY.get("_id_counter", 0)
    SESSION_REGISTRY["_id_counter"] = n + 1
    return f"{prefix}-{n}"


def register_agent(
    agent_id: str,
    parent_id: str | None,
    depth: int,
    is_root: bool,
    topic: str,
) -> None:
    SESSION_REGISTRY[agent_id] = {
        "agent_id": agent_id,
        "parent_id": parent_id,
        "depth": depth,
        "is_root": is_root,
        "topic": topic,
        "status": "in_progress",
        "findings": "",
        "sources": [],
        "iterations": 0,
        "started_at": _elapsed(),
        "completed_at": None,
    }


def update_agent(agent_id: str, **fields) -> None:
    if agent_id in SESSION_REGISTRY:
        SESSION_REGISTRY[agent_id].update(fields)


def build_session_context(for_agent_id: str) -> str:
    """Build the 'Other agents in this session' block for cross-referencing."""
    lines = ["Other agents in this research session:"]
    for aid, entry in SESSION_REGISTRY.items():
        if aid.startswith("_") or aid == for_agent_id:
            continue
        has_findings = "yes" if entry.get("findings") else "no"
        topic = entry.get("topic", "")[:80]
        lines.append(
            f"  - {aid} [{entry.get('status', '?')}] (findings: {has_findings}): {topic}"
        )
    if len(lines) == 1:
        return ""
    lines.append("")
    lines.append(
        "Use reference_findings(agent_id) to reuse another agent's "
        "research instead of re-searching the same topic."
    )
    return "\n".join(lines)


def log_event(agent_id: str, msg: str) -> None:
    entry = SESSION_REGISTRY.get(agent_id, {})
    depth = entry.get("depth", 0)
    indent = "  " * depth
    print(f"[{_elapsed():6.1f}s] {indent}{agent_id:14s} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Tool implementations
#
# LlamaIndex AgentWorkflow invokes tools directly (no manual dispatch).
# Tools that need recursive agent spawning or registry access use module-level
# state. Each tool function must be synchronous for FunctionTool, but we use
# asyncio.run_coroutine_threadsafe / event loop tricks where needed.
# ---------------------------------------------------------------------------

# We store per-invocation context in a dict keyed by asyncio.Task so that
# tool functions can access the calling agent's metadata.
_AGENT_CONTEXT: dict[str, dict] = {}


def _tool_search(query: str, max_results: int = 5) -> str:
    """Search the web for a specific angle or follow-up query.

    Your topic was already searched when you were created and results are in
    your context -- use this only to explore DIFFERENT angles or follow-up
    questions, not to repeat your initial search.
    """
    try:
        result = serper_search(query, max_results=max_results)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _tool_read_pages(urls: list[str]) -> str:
    """Read one or more web pages in parallel.

    Returns each page's content as markdown text (truncated to 4000 chars
    each). Pass ALL the URLs you want to read in a single call -- they are
    fetched simultaneously.
    """
    if not urls:
        return json.dumps({"error": "No URLs provided"})
    fetched = fetch_urls(urls)
    pages = []
    for url in urls:
        content = fetched.get(url)
        if content:
            pages.append({"url": url, "content": content[:4000]})
        else:
            pages.append({"url": url, "error": f"Failed to fetch {url}"})
    return json.dumps({"pages": pages})


def _tool_reference_findings(agent_id: str) -> str:
    """Reference the findings of another agent that has already researched a
    similar or related topic.

    Use this instead of re-searching a topic that another agent has already
    covered. Check the 'Other agents in this research session' block in your
    context to see what topics are available and which have completed.
    """
    ref_entry = SESSION_REGISTRY.get(agent_id)
    if ref_entry and ref_entry.get("findings"):
        return json.dumps(
            {
                "agent_id": agent_id,
                "status": ref_entry.get("status"),
                "findings": ref_entry["findings"],
            }
        )
    return json.dumps({"error": f"Agent {agent_id} not found or has no findings yet."})


def _tool_write_report(report: str) -> str:
    """Write the final research report.

    Call this when you have gathered all findings from your sub-agents and
    any additional research, and are ready to produce the final output.
    """
    # The report content is captured from the return value by the caller.
    return json.dumps({"status": "Report saved", "report": report})


# ---------------------------------------------------------------------------
# Recursive agent runner
# ---------------------------------------------------------------------------


async def run_agent(
    topic: str,
    is_root: bool,
    parent_id: str | None = None,
    depth: int = 0,
) -> dict:
    """Run a single research agent and return its results.

    This function is called recursively by spawn_agents to create the agent
    tree. Each agent gets its own AgentWorkflow instance with the appropriate
    tools and system prompt.
    """
    agent_id = _next_agent_id("root" if is_root else "sub")
    register_agent(agent_id, parent_id, depth, is_root, topic)
    log_event(agent_id, f"created (depth={depth})")

    # Pre-search the topic
    search_context = ""
    try:
        results = serper_search(topic)
        search_context = format_results_for_context(topic, results)
    except Exception as e:
        search_context = f"Initial search failed ({e}). Use the search tool instead."

    # Collected sources and child findings for this agent
    agent_sources: list[dict] = []
    agent_children: list[dict] = []
    agent_report: str | None = None

    # -- spawn_agents tool (defined per-agent to capture depth/agent_id) -----

    def _tool_spawn_agents(queries: list[str]) -> str:
        """Spawn parallel sub-agents to research different topics independently.

        Each sub-agent automatically gets web search results for its topic and can
        then read pages, search for new angles, or spawn its own sub-agents.
        Returns their combined findings when all complete.
        """
        if depth >= MAX_DEPTH:
            return json.dumps(
                {
                    "error": (
                        f"Maximum depth ({MAX_DEPTH}) reached. "
                        "Research this topic directly using search and "
                        "read_pages instead."
                    )
                }
            )
        if not queries:
            return json.dumps({"error": "No queries provided"})

        log_event(
            agent_id,
            f"spawn {len(queries)} children: {[q[:30] for q in queries]}",
        )

        # Run child agents concurrently. We need to get back to the event
        # loop from inside a sync tool call. LlamaIndex runs sync tools in
        # threads, so we schedule the coroutines on the running event loop.
        loop = asyncio.get_event_loop()
        child_coros = [
            run_agent(
                topic=q,
                is_root=False,
                parent_id=agent_id,
                depth=depth + 1,
            )
            for q in queries
        ]
        futures = [asyncio.run_coroutine_threadsafe(c, loop) for c in child_coros]
        child_results = [f.result() for f in futures]

        for child in child_results:
            agent_children.append(child)
            agent_sources.extend(child.get("sources", []))

        compiled = [
            {
                "agent_id": child.get("agent_id"),
                "topic": child["topic"],
                "findings": child.get("findings") or "(no findings)",
                "verified_sources": child.get("sources", []),
            }
            for child in child_results
        ]
        return json.dumps({"sub_agent_results": compiled})

    # -- Wrap read_pages to track sources ---------------------------------

    def _tracking_read_pages(urls: list[str]) -> str:
        """Read one or more web pages in parallel.

        Returns each page's content as markdown text (truncated to 4000 chars
        each). Pass ALL the URLs you want to read in a single call -- they are
        fetched simultaneously.
        """
        result_str = _tool_read_pages(urls)
        # Track sources from successfully fetched pages
        try:
            data = json.loads(result_str)
            for page in data.get("pages", []):
                if "content" in page:
                    title = page["content"][:100].split("\n")[0]
                    agent_sources.append({"url": page["url"], "title": title})
        except Exception:
            pass
        return result_str

    # -- Wrap write_report to capture ----------------------------------------

    def _tracking_write_report(report: str) -> str:
        """Write the final research report.

        Call this when you have gathered all findings from your sub-agents and
        any additional research, and are ready to produce the final output.
        """
        nonlocal agent_report
        agent_report = report
        update_agent(
            agent_id,
            status="completed",
            findings=report,
            completed_at=_elapsed(),
        )
        log_event(agent_id, "write_report")
        return json.dumps({"status": "Report saved"})

    # -- Build tools list ----------------------------------------------------

    tools = [
        FunctionTool.from_defaults(fn=_tool_search, name="search"),
        FunctionTool.from_defaults(fn=_tracking_read_pages, name="read_pages"),
        FunctionTool.from_defaults(fn=_tool_spawn_agents, name="spawn_agents"),
        FunctionTool.from_defaults(fn=_tool_reference_findings, name="reference_findings"),
    ]
    if is_root:
        tools.append(
            FunctionTool.from_defaults(fn=_tracking_write_report, name="write_report")
        )

    system_prompt = ROOT_AGENT_SYSTEM if is_root else SUB_AGENT_SYSTEM

    # Inject pre-search results and session context into the system prompt
    session_ctx = build_session_context(agent_id)
    full_system = system_prompt
    if search_context:
        full_system += f"\n\nInitial search results for your topic:\n\n{search_context}"
    if session_ctx:
        full_system += f"\n\n{session_ctx}"

    agent = AgentWorkflow(
        agents=[
            FunctionAgent(
                name="Agent",
                description="A research agent.",
                tools=tools,
                llm=llm,
                system_prompt=full_system,
                streaming=False,
            )
        ],
    )

    user_text = (
        f"Research the following topic and produce a comprehensive report: {topic}"
        if is_root
        else f"Research the following topic thoroughly: {topic}"
    )

    log_event(agent_id, "running")
    response = await agent.run(user_text)
    response_text = str(response)
    log_event(agent_id, "finished")

    # If the agent didn't explicitly write findings, use its final response
    if not SESSION_REGISTRY[agent_id].get("findings"):
        update_agent(
            agent_id,
            status="completed",
            findings=response_text,
            completed_at=_elapsed(),
        )

    return {
        "agent_id": agent_id,
        "parent_id": parent_id,
        "topic": topic,
        "is_root": is_root,
        "depth": depth,
        "findings": agent_report or response_text,
        "report": agent_report,
        "sources": agent_sources,
        "children": agent_children,
    }


# ---------------------------------------------------------------------------
# Synthesis fallback
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT = """\
All research is now complete. Based on all the findings below, write a \
comprehensive, well-structured research report in markdown. Include an \
executive summary, thematic sections with source citations, areas where \
sources disagree, and areas for further research.

CITATION RULES:
- Only cite URLs from the verified sources list below.
- Do not cite URLs from search snippets or invent URLs.
- If a finding has no verified URL, state it without a link.

Output ONLY the report -- no preamble or commentary."""


async def run_research(topic: str) -> dict:
    """Run a full research session with force-complete + synthesis fallback."""
    _reset_session()

    print(f"Starting research: {topic}")
    print(f"Model: {MODEL}")
    print()

    result = await run_agent(topic=topic, is_root=True)

    # Force-complete: any agent still in_progress hit max_iterations.
    for aid, entry in SESSION_REGISTRY.items():
        if aid.startswith("_"):
            continue
        if entry.get("status") == "in_progress":
            entry["status"] = "incomplete"
            if not entry.get("findings"):
                entry["findings"] = "Max iterations reached before completion."
            entry["completed_at"] = _elapsed()

    # If the root has no report, do one final synthesis round.
    if not result.get("report"):
        print()
        print("Root did not call write_report; running synthesis fallback...")

        # De-dupe sources from the entire tree.
        all_sources = list({s["url"]: s for s in result.get("sources", [])}.values())
        sources_block = ""
        if all_sources:
            source_lines = [f"- [{s['title']}]({s['url']})" for s in all_sources]
            sources_block = (
                "\n\nVERIFIED SOURCES -- these URLs were actually fetched and "
                "read during research. Use ONLY these for citations:\n"
                + "\n".join(source_lines)
            )

        # Gather all findings into a synthesis prompt
        all_findings = []
        for aid, entry in SESSION_REGISTRY.items():
            if aid.startswith("_"):
                continue
            if entry.get("findings"):
                all_findings.append(
                    f"## {aid}: {entry['topic']}\n\n{entry['findings']}"
                )

        findings_text = "\n\n---\n\n".join(all_findings)
        synthesis_input = (
            f"# Research Findings\n\n{findings_text}\n\n"
            f"---\n\n{SYNTHESIS_PROMPT}{sources_block}"
        )

        synthesis_response = await llm.acomplete(synthesis_input)
        result["report"] = str(synthesis_response)
        result["findings"] = result["report"]
        root_id = result["agent_id"]
        update_agent(
            root_id,
            status="completed",
            findings=result["report"],
            completed_at=_elapsed(),
        )

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def print_tree() -> None:
    """Walk the registry and print the full agent tree."""
    children_by_parent: dict[str | None, list[str]] = {}
    for aid, entry in SESSION_REGISTRY.items():
        if aid.startswith("_"):
            continue
        children_by_parent.setdefault(entry.get("parent_id"), []).append(aid)

    STATUS_ICON = {
        "in_progress": "o",
        "completed": "*",
        "failed": "x",
        "incomplete": "-",
    }

    def _walk(aid: str, prefix: str, is_last: bool) -> None:
        entry = SESSION_REGISTRY[aid]
        connector = "`- " if is_last else "|- "
        icon = STATUS_ICON.get(entry.get("status", "?"), "?")
        topic = entry.get("topic", "")[:60]
        elapsed = entry.get("completed_at") or 0
        print(f"  {prefix}{connector}[{icon}] {aid} ({elapsed:.1f}s) {topic}")
        children = children_by_parent.get(aid, [])
        new_prefix = prefix + ("   " if is_last else "|  ")
        for i, cid in enumerate(children):
            _walk(cid, new_prefix, i == len(children) - 1)

    roots = children_by_parent.get(None, [])
    for rid in roots:
        _walk(rid, "", True)


def write_session_files(result: dict, out_dir: Path) -> None:
    """Dump agent-tree.json + summary.json + report.md into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    agents = [
        entry for aid, entry in SESSION_REGISTRY.items() if not aid.startswith("_")
    ]

    with open(out_dir / "agent-tree.json", "w") as f:
        json.dump(agents, f, indent=2)

    counts: dict[str, int] = {}
    for entry in agents:
        status = entry.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    summary = {
        "topic": TOPIC,
        "model": MODEL,
        "total_agents": len(agents),
        "by_status": counts,
        "max_depth": max((a.get("depth", 0) for a in agents), default=0),
        "elapsed_seconds": _elapsed(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    report = result.get("report") or result.get("findings") or "(no output)"
    with open(out_dir / "report.md", "w") as f:
        f.write(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    result = await run_research(TOPIC)

    print()
    print("=" * 60)
    print(f"Topic: {TOPIC}")
    agent_count = len([k for k in SESSION_REGISTRY if not k.startswith("_")])
    print(f"Total agents: {agent_count}")
    print(f"Sources collected: {len(result.get('sources', []))}")
    print(f"Elapsed: {_elapsed():.1f}s")
    print("=" * 60)
    print()
    print_tree()

    write_session_files(result, OUTPUT_DIR)
    print(f"\nWrote {OUTPUT_DIR}/report.md, agent-tree.json, summary.json")

    print()
    print("=" * 60)
    print("REPORT")
    print("=" * 60)
    report_text = result.get("report") or result.get("findings") or "(no output)"
    print(report_text)


if __name__ == "__main__":
    asyncio.run(main())
