"""
Verifier agent — checks claims against sources / graph.
"""


def verify_claim(claim: dict, context: dict | None = None) -> dict:
    """Verify a single claim. Returns verdict + metadata (stub)."""
    # TODO: use graph + causal checks
    return {"verified": False, "confidence": 0.0}
