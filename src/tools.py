"""
Search tools — web_search (general), x_keyword_search (X-specific).
Mock: return synthetic similar reports. Replace with real APIs later.
"""
from typing import Callable

# Mock responses for common claim types (for demo without real API)
MOCK_REPORTS = {
    "conspiracy": [
        "Fact-check: No evidence vaccines contain microchips. FDA and CDC deny.",
        "Debunked: 5G towers do not cause COVID-19. WHO statement.",
        "False: Moon landing was not faked. NASA archives confirm.",
    ],
    "earthquake": [
        "USGS: No 7.2 magnitude earthquake in Nashville today.",
        "Local news: Nashville area seismic activity normal.",
        "Emergency services: No earthquake reported in Tennessee.",
    ],
    "default": [
        "Multiple fact-checkers have reviewed similar claims.",
        "No corroborating evidence found in mainstream sources.",
        "Claim lacks verification from reliable outlets.",
    ],
}


def _classify_query(statement: str) -> str:
    """Simple mock: classify query type for mock responses."""
    s = statement.lower()
    if any(w in s for w in ["conspiracy", "vaccine", "5g", "moon", "alien", "flat"]):
        return "conspiracy"
    if any(w in s for w in ["earthquake", "magnitude", "nashville"]):
        return "earthquake"
    return "default"


def web_search(query: str, max_results: int = 3) -> list[str]:
    """
    General web search. Mock: returns synthetic similar reports.
    Replace with real API (e.g. Serper, Tavily) later.
    """
    bucket = _classify_query(query)
    reports = MOCK_REPORTS.get(bucket, MOCK_REPORTS["default"])
    return reports[:max_results]


def x_keyword_search(query: str, max_results: int = 3) -> list[str]:
    """
    X (Twitter) keyword search for similar posts. Mock: same as web_search.
    Replace with X API later.
    """
    return web_search(query, max_results)


# Optional: pluggable real implementations
_web_search_impl: Callable[[str, int], list[str]] = web_search
_x_search_impl: Callable[[str, int], list[str]] = x_keyword_search


def set_web_search(fn: Callable[[str, int], list[str]]) -> None:
    global _web_search_impl
    _web_search_impl = fn


def set_x_search(fn: Callable[[str, int], list[str]]) -> None:
    global _x_search_impl
    _x_search_impl = fn
