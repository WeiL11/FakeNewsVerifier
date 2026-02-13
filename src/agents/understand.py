"""
Understand Model — Parses input claim into a clean "statement".
Mock: regex/NLP heuristics to extract key facts. Later tunable via RL or fine-tuning.
"""
import re


# Patterns that often indicate hedging or filler
FILLER_PATTERNS = [
    r"\b(it has been reported|sources say|allegedly|reportedly)\b",
    r"\b(according to|as per|it is said that)\b",
    r"\b(breaking|urgent|exclusive)\s*:?\s*",
]

# Extract quoted or emphasized phrases
QUOTE_PATTERN = r'"([^"]+)"|«([^»]+)»'


def understand(claim: str) -> dict:
    """
    Parse input claim into a clean statement and extracted facts.
    Returns: {"statement": str, "key_facts": list[str], "raw": str}
    """
    raw = claim.strip()
    if not raw:
        return {"statement": "", "key_facts": [], "raw": raw}

    # Remove filler phrases (mock NLP)
    statement = raw
    for pat in FILLER_PATTERNS:
        statement = re.sub(pat, "", statement, flags=re.IGNORECASE)
    statement = re.sub(r"\s+", " ", statement).strip()

    # Extract quoted/key phrases as "facts"
    key_facts = []
    for m in re.finditer(QUOTE_PATTERN, raw):
        key_facts.append((m.group(1) or m.group(2) or "").strip())
    if not key_facts:
        # Fallback: split on sentence boundaries, take first clause
        parts = re.split(r"[.!?]\s+", statement)
        if parts:
            key_facts = [p.strip() for p in parts[:2] if len(p) > 10]

    return {
        "statement": statement or raw,
        "key_facts": key_facts[:5],
        "raw": raw,
    }
