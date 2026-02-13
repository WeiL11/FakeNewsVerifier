"""
Causal intervention: adjusts confidence via SymPy model.
Params (bias, truth) are tunable by RL.
"""
from sympy import symbols, Eq, solve


def simple_causal_intervention(claim, assumed_cause="source_bias", bias=0.4, truth=0.85):
    """Model confidence = truth - bias; return adjusted confidence and intervention label."""
    b, t, conf = symbols("bias truth confidence")
    eq = Eq(conf, t - b)
    solution = solve(eq.subs({b: bias, t: truth}), conf)
    adjusted = float(solution[0]) if solution else 0.5

    return {
        "original_confidence": 0.6,
        "adjusted_confidence": round(min(1.0, max(0.0, adjusted)), 2),
        "intervention": f"Removed {assumed_cause}",
    }
