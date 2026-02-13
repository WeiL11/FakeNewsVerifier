"""
Claim–source graph (NetworkX) for reasoning over claims and sources.
"""
import networkx as nx


def build_claim_graph(claims, sources=None):
    """
    Simple directed graph: claims point to sources/evidence.
    Returns graph + basic stats.
    """
    G = nx.DiGraph()

    if sources is None:
        sources = ["Web search mock", "Known fact DB", "User post"]

    for i, claim in enumerate(claims):
        label = f"{claim[:40]}..." if len(claim) > 40 else claim
        claim_node = f"claim_{i}: {label}"
        G.add_node(claim_node, type="claim", text=claim, confidence=0.0)

        for src in sources:
            src_node = f"src_{src}"
            G.add_node(src_node, type="source")
            G.add_edge(claim_node, src_node, weight=0.7)

    return G


def graph_summary(G):
    """Simple text summary of the graph."""
    claims = [n for n, d in G.nodes(data=True) if d.get("type") == "claim"]
    sources = len([n for n, d in G.nodes(data=True) if d.get("type") == "source"])
    edges = G.number_of_edges()
    return f"Graph: {len(claims)} claims • {sources} sources • {edges} edges"
