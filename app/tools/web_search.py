"""Live web search fallback (Tavily) for questions the uploaded PDFs can't answer."""

import requests

from app.config import TAVILY_API_KEY

SEARCH_URL = "https://api.tavily.com/search"


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Query Tavily for current web results. Returns [] if no API key is configured."""
    if not TAVILY_API_KEY:
        return []

    response = requests.post(
        SEARCH_URL,
        json={
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        },
        timeout=15,
    )
    response.raise_for_status()
    results = response.json().get("results", [])

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in results
    ]
