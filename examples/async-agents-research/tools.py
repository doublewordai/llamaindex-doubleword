"""Web search and page-reading tools for research agents.

- ``search``: Serper API wrapper for web search.
- ``fetch_urls``: Jina Reader API wrapper for fetching page content as markdown.
- ``format_results_for_context``: Format search results for injection into agent
  messages.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Serper search
# ---------------------------------------------------------------------------


def _get_serper_key() -> str:
    """Get Serper API key from environment."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable not set")
    return api_key


def search(query: str, max_results: int = 10) -> dict:
    """Run a Serper web search.

    Args:
        query: Search query string.
        max_results: Maximum number of results (default 10).

    Returns:
        Dict with ``results`` list containing ``url``, ``title``, ``snippet``.
    """
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": _get_serper_key()},
        json={"q": query, "num": max_results},
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic", []):
        results.append(
            {
                "url": item.get("link"),
                "title": item.get("title"),
                "snippet": item.get("snippet"),
            }
        )
    return {"results": results}


def format_results_for_context(query: str, results: dict) -> str:
    """Format search results as readable text for injection into agent messages."""
    items = results.get("results", [])
    if not items:
        return f'Search for "{query}" returned no results.'

    lines = [f'Search results for "{query}":\n']
    for i, item in enumerate(items, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        lines.append(f"{i}. [{title}]({url})")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Jina Reader (page fetching)
# ---------------------------------------------------------------------------


def _fetch_single_url(url: str, timeout: int = 15) -> Optional[str]:
    """Fetch a single URL via Jina Reader API (HTML -> markdown)."""
    try:
        response = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain"},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text[:50000]
    except Exception:
        return None


def fetch_urls(urls: list[str], max_workers: int = 5) -> dict[str, Optional[str]]:
    """Fetch multiple URLs in parallel via Jina Reader.

    Args:
        urls: List of URLs to fetch.
        max_workers: Number of parallel workers.

    Returns:
        Dict mapping URL -> markdown content (or ``None`` if failed).
    """
    results: dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(_fetch_single_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = None
    return results
