"""
Understand: parses claim into clean statement and key facts.
Uses regex heuristics; replace with NLP/LLM for production.
"""
import re

# Hedging/filler patterns to strip
FILLER_PATTERNS = [
    r"\b(it has been reported|sources say|allegedly|reportedly)\b",
    r"\b(according to|as per|it is said that)\b",
    r"\b(breaking|urgent|exclusive)\s*:?\s*",
]

QUOTE_PATTERN = r'"([^"]+)"|«([^»]+)»'


def understand(claim: str) -> dict:
    """Parse claim into statement and key_facts. Returns {statement, key_facts, raw}."""
    raw = claim.strip()
    if not raw:
        return {"statement": "", "key_facts": [], "raw": raw}

    statement = raw
    for pat in FILLER_PATTERNS:
        statement = re.sub(pat, "", statement, flags=re.IGNORECASE)
    statement = re.sub(r"\s+", " ", statement).strip()

    key_facts = []
    for m in re.finditer(QUOTE_PATTERN, raw):
        key_facts.append((m.group(1) or m.group(2) or "").strip())
    if not key_facts:
        parts = re.split(r"[.!?]\s+", statement)
        if parts:
            key_facts = [p.strip() for p in parts[:2] if len(p) > 10]

    return {
        "statement": statement or raw,
        "key_facts": key_facts[:5],
        "raw": raw,
    }
