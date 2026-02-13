"""
Corrector agent — self-correction for low-confidence claims via causal intervention.
Threshold and causal params are tunable (e.g. by RL).
"""
import copy

from src.causal import simple_causal_intervention

DEFAULT_PARAMS = {
    "intervention_threshold": 0.55,
    "causal_bias": 0.4,
    "causal_truth": 0.85,
}


def correct_results(verification_results, graph, params=None):
    """
    Self-correction loop: for low-confidence claims, apply causal intervention.
    If conf < intervention_threshold → intervene once.
    params: optional dict from RL tuner (intervention_threshold, causal_bias, causal_truth).
    Fallback: if params is None, use DEFAULT_PARAMS (hardcoded).
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    threshold = params["intervention_threshold"]  # learned_value when from tuner
    bias = params["causal_bias"]
    truth = params["causal_truth"]

    corrected = []
    for item in copy.deepcopy(verification_results):
        if item["confidence"] < threshold:
            intervention = simple_causal_intervention(
                item["claim"], bias=bias, truth=truth
            )
            item["confidence"] = intervention["adjusted_confidence"]
            item["status"] = (
                "High" if item["confidence"] >= 0.75
                else "Medium" if item["confidence"] >= 0.55
                else "Low (corrected)"
            )
            item["correction_applied"] = intervention["intervention"]
        else:
            item["correction_applied"] = None
        corrected.append(item)

    return corrected, graph
