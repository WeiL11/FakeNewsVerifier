"""
Search tools: web_search and x_keyword_search.
Mock implementations return synthetic reports. Replace with real APIs (Serper, Tavily, X API) for production.
"""

# Mock responses by claim type (demo only)
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
    """Classify query type to select mock response bucket."""
    s = statement.lower()
    if any(w in s for w in ["conspiracy", "vaccine", "5g", "moon", "alien", "flat"]):
        return "conspiracy"
    if any(w in s for w in ["earthquake", "magnitude", "nashville"]):
        return "earthquake"
    return "default"


def web_search(query: str, max_results: int = 3) -> list[str]:
    """General web search. Returns synthetic reports for demo; replace with real API for production."""
    bucket = _classify_query(query)
    reports = MOCK_REPORTS.get(bucket, MOCK_REPORTS["default"])
    return reports[:max_results]


def x_keyword_search(query: str, max_results: int = 3) -> list[str]:
    """X (Twitter) keyword search. Mock: delegates to web_search; replace with X API for production."""
    return web_search(query, max_results)
