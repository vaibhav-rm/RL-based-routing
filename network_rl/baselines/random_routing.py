"""
Random (flooding-style) routing baseline.

CN context:
- Pure flooding sends a packet out ALL interfaces; here we simulate a
  simplified version that picks a random available next-hop.
- This mimics early ARPANet behaviour and represents the worst-case
  intelligent baseline — any learned policy should significantly outperform it.
- Loop-free flooding uses sequence numbers; we use visited-set tracking here.
"""

import random
from typing import Optional, List
import networkx as nx


def random_next_hop(
    graph: nx.Graph,
    source: int,
    destination: int,
    visited: Optional[set] = None,
) -> Optional[int]:
    """
    Pick a random active neighbour of source, preferring unvisited nodes
    to avoid infinite loops (analogous to the TTL field in IP packets).

    Returns None if no valid next-hop exists (dead end).
    """
    if source == destination:
        return destination

    neighbors = [n for n in graph.neighbors(source)]
    if not neighbors:
        return None

    # Prefer unvisited neighbours to reduce looping
    if visited:
        unvisited = [n for n in neighbors if n not in visited]
        if unvisited:
            return random.choice(unvisited)

    return random.choice(neighbors)


def random_full_path(
    graph: nx.Graph,
    source: int,
    destination: int,
    max_hops: int = 20,
) -> List[int]:
    """
    Walk from source to destination via random hops.
    Returns the path taken (may be empty if destination unreachable within max_hops).
    """
    path = [source]
    visited = {source}
    current = source

    for _ in range(max_hops):
        if current == destination:
            break
        nxt = random_next_hop(graph, current, destination, visited)
        if nxt is None:
            break
        path.append(nxt)
        visited.add(nxt)
        current = nxt

    return path if path[-1] == destination else path
