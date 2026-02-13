"""
Simple causal checks using SymPy (e.g. intervention, adjusted confidence).
Params (bias, truth) are tunable by RL.
"""
from sympy import symbols, Eq, solve


def simple_causal_intervention(claim, assumed_cause="source_bias", bias=0.4, truth=0.85):
    """
    Toy causal check: model claim as C = f(Bias, Truth)
    Simulate 'do(remove bias)' → adjusted confidence.
    bias, truth: tunable (e.g. by RL).
    """
    b, t, conf = symbols("bias truth confidence")
    eq = Eq(conf, t - b)
    solution = solve(eq.subs({b: bias, t: truth}), conf)
    adjusted = float(solution[0]) if solution else 0.5

    return {
        "original_confidence": 0.6,
        "adjusted_confidence": round(min(1.0, max(0.0, adjusted)), 2),
        "intervention": f"Removed {assumed_cause}",
    }
