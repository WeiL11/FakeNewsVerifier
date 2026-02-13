"""
Claim–source graph (networkx) for reasoning over claims and sources.
"""
import networkx as nx
from typing import Any


def build_claim_source_graph(claims: list[dict], sources: list[dict] | None = None) -> nx.DiGraph:
    """Build a directed graph linking claims to sources (stub)."""
    G = nx.DiGraph()
    # TODO: add nodes (claims, sources) and edges
    return G
