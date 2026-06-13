"""
Dijkstra's shortest-path routing baseline.

CN context:
- OSPF (Open Shortest Path First) uses Dijkstra on the link-state database.
- Edge weight here = effective delay after congestion, mirroring how OSPF
  metric = interface cost × (1 + congestion factor).
- This is the gold-standard comparison: DQN should learn to match or beat
  Dijkstra under dynamic / failure conditions where global state is imperfect.
"""

import networkx as nx
from typing import Optional, List


def dijkstra_next_hop(
    graph: nx.Graph,
    source: int,
    destination: int,
    weight_attr: str = "weight",
) -> Optional[int]:
    """
    Return the first hop on the shortest path from source to destination.
    Returns None if destination is unreachable (all paths failed).

    The graph edges must carry a 'weight' attribute representing effective cost
    (base_delay × congestion_factor). Inactive links should be removed or have
    infinite weight before calling this function.
    """
    if source == destination:
        return destination

    try:
        path: List[int] = nx.dijkstra_path(graph, source, destination, weight=weight_attr)
        return path[1] if len(path) > 1 else destination
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def dijkstra_full_path(
    graph: nx.Graph,
    source: int,
    destination: int,
    weight_attr: str = "weight",
) -> List[int]:
    """Return the complete shortest path (list of nodes) or empty list if unreachable."""
    try:
        return nx.dijkstra_path(graph, source, destination, weight=weight_attr)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def build_weighted_graph(env) -> nx.Graph:
    """
    Build a NetworkX graph from the environment, weighting edges by
    effective delay (base_delay × (1 + 3 × congestion)).
    Failed links are excluded — Dijkstra cannot route over a down link.
    """
    H = nx.Graph()
    H.add_nodes_from(env.G.nodes())
    for (u, v) in env.edge_list:
        edge = env.G.edges[u, v]
        if edge["active"]:
            effective_delay = edge["base_delay"] * (1.0 + 3.0 * edge["congestion"])
            H.add_edge(u, v, weight=effective_delay)
    return H
