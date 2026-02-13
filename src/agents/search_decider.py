"""
Search Decider: outputs search_need (0–1).
High need when fake_conf or conflict exceeds thresholds.
"""
from typing import Any


def search_decide(
    fake_conf: float,
    common_sense_conflict: float,
    params: dict[str, Any] | None = None,
) -> dict:
    """Return search_need based on fake_conf and conflict thresholds."""
    params = params or {}
    fake_thresh = params.get("fake_threshold", 0.5)
    conflict_thresh = params.get("conflict_threshold", 0.5)

    need = 0.0
    if fake_conf >= fake_thresh:
        need = max(need, min(1.0, 0.5 + fake_conf))
    if common_sense_conflict >= conflict_thresh:
        need = max(need, min(1.0, 0.5 + common_sense_conflict))

    if fake_conf >= 0.6 and common_sense_conflict >= 0.6:
        need = min(1.0, need + 0.2)

    return {
        "search_need": round(min(1.0, max(0.0, need)), 3),
        "fake_conf": fake_conf,
        "common_sense_conflict": common_sense_conflict,
    }
