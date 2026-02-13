"""
Reporter — Outputs final score + matrix for before/after observation.
"""
from typing import Any


def report(
    original_claim: str,
    understand_result: dict,
    judge_result: dict,
    search_need: float,
    search_triggered: bool,
    searched_reports: list[str],
    searched_scores: list[dict],
    comparator_result: dict,
) -> dict:
    """
    Build final report with score matrix for before/after observation.
    """
    final_is_fake = comparator_result.get("final_is_fake", judge_result.get("fake_conf", 0.5))

    matrix = {
        "understand": {
            "statement": understand_result.get("statement", ""),
            "key_facts": understand_result.get("key_facts", []),
        },
        "judge": {
            "fake_conf": judge_result.get("fake_conf", 0),
            "common_sense_conflict": judge_result.get("common_sense_conflict", 0),
        },
        "search_decider": search_need,
        "search_triggered": search_triggered,
        "searched_reports": searched_reports,
        "searched_scores": searched_scores,
        "comparator": comparator_result,
        "final_is_fake": final_is_fake,
    }

    return {
        "original_claim": original_claim,
        "final_is_fake": final_is_fake,
        "matrix": matrix,
        "summary": _format_summary(matrix),
    }


def _format_summary(matrix: dict) -> str:
    j = matrix.get("judge", {})
    c = matrix.get("comparator", {})
    triggered = matrix.get("search_triggered", False)
    final = matrix.get("final_is_fake", 0)
    status = "FAKE" if final > 0.5 else "LIKELY REAL"
    parts = [
        f"Final: {final:.2f} ({status})",
        f"Judge: fake={j.get('fake_conf', 0):.2f}, conflict={j.get('common_sense_conflict', 0):.2f}",
    ]
    if triggered:
        parts.append(f"Search: {c.get('num_reports', 0)} reports, aggregated fake={c.get('aggregated_fake_conf', 0):.2f}")
    else:
        parts.append("Search: not triggered")
    return " | ".join(parts)
