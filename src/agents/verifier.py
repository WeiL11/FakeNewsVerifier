"""
Verifier: builds claim–source graph and assigns confidence scores.
"""
from src.graph import build_claim_graph, graph_summary


def verify_claims(claims):
    """Build graph and assign confidence per claim. Replace with real search/KB for production."""
    graph = build_claim_graph(claims)

    results = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "claim":
            conf = 0.4 + 0.5 * (len(data["text"]) % 10) / 10
            graph.nodes[node]["confidence"] = round(conf, 2)
            results.append({
                "claim": data["text"],
                "confidence": round(conf, 2),
                "status": (
                    "High" if conf >= 0.75
                    else "Medium" if conf >= 0.55
                    else "Low"
                ),
            })

    return results, graph
