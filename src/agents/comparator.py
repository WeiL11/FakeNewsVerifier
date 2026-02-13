"""
Comparator: blends original judge scores with aggregated scores from searched reports.
Outputs final is_fake (0–1).
"""
from typing import Any


def compare(
    original_fake_conf: float,
    original_conflict: float,
    searched_scores: list[dict],
    params: dict[str, Any] | None = None,
) -> dict:
    """Weighted blend of original and aggregated fake_conf. Returns final_is_fake and metadata."""
    params = params or {}
    weight_original = params.get("weight_original", 0.5)
    weight_searched = params.get("weight_searched", 0.5)

    if not searched_scores:
        return {
            "final_is_fake": round(original_fake_conf, 3),
            "original_fake_conf": original_fake_conf,
            "aggregated_fake_conf": original_fake_conf,
            "num_reports": 0,
        }

    agg_fake = sum(s.get("fake_conf", 0.5) for s in searched_scores) / len(searched_scores)
    agg_conflict = sum(s.get("common_sense_conflict", 0.5) for s in searched_scores) / len(searched_scores)

    final = weight_original * original_fake_conf + weight_searched * agg_fake
    final = min(1.0, max(0.0, final))

    return {
        "final_is_fake": round(final, 3),
        "original_fake_conf": original_fake_conf,
        "aggregated_fake_conf": round(agg_fake, 3),
        "aggregated_conflict": round(agg_conflict, 3),
        "num_reports": len(searched_scores),
    }
