"""
ECMP — Equal-Cost Multi-Path routing.

CN context:
  ECMP is the production standard in data-centre and ISP backbones.
  When multiple shortest paths exist (equal IGP cost), traffic is
  hashed across all of them (per-flow or per-packet).
  RFC 2991 / RFC 2992 define ECMP behaviour.

  This baseline is harder to beat than plain Dijkstra because it
  spreads load and provides resilience when multiple equal-cost
  paths exist — exactly the condition our topology was designed for.
"""

import networkx as nx
import hashlib
from typing import List, Optional, Tuple


def _all_shortest_paths(graph: nx.Graph, source: int, dest: int,
                         weight: str = "weight") -> List[List[int]]:
    """Return all shortest paths (same cost) between source and dest."""
    try:
        paths = list(nx.all_shortest_paths(graph, source, dest, weight=weight))
        return paths
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def ecmp_next_hop(
    graph:  nx.Graph,
    source: int,
    dest:   int,
    flow_id: int = 0,
    weight: str  = "weight",
) -> Optional[int]:
    """
    Select next-hop via ECMP with per-flow hash (5-tuple hash approximation).
    Tie-breaking uses a deterministic hash of flow_id so the same flow
    always takes the same path (required for TCP in-order delivery).
    """
    if source == dest:
        return dest

    paths = _all_shortest_paths(graph, source, dest, weight=weight)
    if not paths:
        return None

    # Hash flow_id modulo number of equal-cost paths
    idx = flow_id % len(paths)
    path = paths[idx]
    return path[1] if len(path) > 1 else dest


def ecmp_full_path(
    graph:   nx.Graph,
    source:  int,
    dest:    int,
    flow_id: int = 0,
    weight:  str = "weight",
) -> List[int]:
    """Return the full ECMP-selected path."""
    if source == dest:
        return [source]
    paths = _all_shortest_paths(graph, source, dest, weight=weight)
    if not paths:
        return []
    return paths[flow_id % len(paths)]


def ecmp_load_balanced_path(
    graph:  nx.Graph,
    source: int,
    dest:   int,
    weight: str = "weight",
) -> List[int]:
    """
    Among all shortest paths, pick the one whose maximum-link-congestion
    is lowest. This approximates weighted ECMP (W-ECMP) where weights
    are inversely proportional to link load.
    """
    paths = _all_shortest_paths(graph, source, dest, weight=weight)
    if not paths:
        return []

    def max_cong(path):
        mc = 0.0
        for u, v in zip(path[:-1], path[1:]):
            if graph.has_edge(u, v):
                mc = max(mc, graph.edges[u, v].get("congestion", 0.0))
        return mc

    return min(paths, key=max_cong)
