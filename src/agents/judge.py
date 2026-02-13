"""
Judge: outputs fake_conf and common_sense_conflict (0–1).
Uses keyword heuristics; tunable via RL for production.
"""
import re

# Conspiracy/misinformation indicators
CONSPIRACY_WORDS = [
    "conspiracy", "hidden", "cover-up", "suppressed", "truth", "lying",
    "microchips", "5g", "vaccines", "aliens", "pyramids", "flat", "rigged",
    "faked", "hoax", "big pharma", "government hiding", "nasa", "secret",
    "they don't want you to know", "wake up", "sheep", "mainstream media",
]

# Common-sense conflict phrases
CONFLICT_PHRASES = [
    "earth is flat", "moon landing.*faked", "sun revolves", "bleach cures",
    "microchip.*vaccine|vaccine.*microchip", "5g.*covid|covid.*5g", "aliens built",
    "cheese.*moon", "10% of brain", "gum.*7 years", "napoleon.*short", "viking.*horn",
]

# Credibility boosters
CREDIBLE_SIGNALS = [
    "politifact", "fact-check", "reuters", "ap news", "confirmed",
    "official", "announced", "reported", "according to records",
]


def judge(statement: str, key_facts: list[str] | None = None) -> dict:
    """Return fake_conf and common_sense_conflict from keyword heuristics."""
    text = (statement or "").lower()
    facts_text = " ".join(key_facts or []).lower()
    combined = f"{text} {facts_text}"

    fake_score = 0.0
    conflict_score = 0.0

    for w in CONSPIRACY_WORDS:
        if w in combined:
            fake_score = min(1.0, fake_score + 0.15)
            conflict_score = min(1.0, conflict_score + 0.1)

    for pat in CONFLICT_PHRASES:
        if re.search(pat, combined):
            conflict_score = min(1.0, conflict_score + 0.45)
            fake_score = min(1.0, fake_score + 0.4)

    for s in CREDIBLE_SIGNALS:
        if s in combined:
            fake_score = max(0.0, fake_score - 0.2)
            conflict_score = max(0.0, conflict_score - 0.1)

    if len(statement) < 50 and ("!" in statement or "breaking" in statement.lower()):
        fake_score = min(1.0, fake_score + 0.1)

    return {
        "fake_conf": round(min(1.0, max(0.0, fake_score)), 3),
        "common_sense_conflict": round(min(1.0, max(0.0, conflict_score)), 3),
        "statement": statement,
    }
