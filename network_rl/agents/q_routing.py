"""
Q-Routing — Boyan & Littman, 1994.
"Packet Routing in Dynamically Changing Networks: A Reinforcement Learning Approach"
NeurIPS 1994.

This is the foundational RL routing algorithm and a critical research baseline.

Algorithm (decentralised — each router is an independent agent):
  Each router k stores Q_k[d][n] = estimated delivery time from k to d via next-hop n.

  On receiving a packet destined for d via neighbour n_prev:
    The packet carries q̂_n = min_{n'} Q_n[d][n']  (best estimate at previous hop)
    Update: Q_k[d][n_prev] += η · (q_k + q̂_n - Q_k[d][n_prev])
    where q_k = queue delay experienced at node k (estimated from congestion)

Key properties vs DQN:
  ✓ Decentralised (no global state)
  ✓ Online, incremental updates
  ✗ Slow convergence (propagates info one hop per packet)
  ✗ Cannot handle topological changes quickly
  ✗ No function approximation → O(nodes × destinations) table
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple


class QRoutingAgent:
    """
    Implements Q-routing for the full network.
    Maintains Q-tables for ALL nodes (simulates distributed deployment).

    self.Q[k][d][n] = estimated delivery time from node k to dest d via neighbour n
    """

    def __init__(
        self,
        graph: nx.Graph,
        lr:          float = 0.1,
        init_val:    float = 10.0,   # initial Q value (optimistic)
    ):
        self.graph    = graph
        self.lr       = lr
        self.num_nodes = graph.number_of_nodes()

        # Initialise Q-tables: Q[k][d] = {n: Q_value}
        self.Q: Dict[int, Dict[int, Dict[int, float]]] = {}
        for k in graph.nodes():
            self.Q[k] = {}
            for d in graph.nodes():
                if d == k:
                    continue
                neighbors = sorted(graph.neighbors(k))
                self.Q[k][d] = {n: init_val for n in neighbors}

    # ── routing decision ────────────────────────────────────────────

    def select_next_hop(self, node: int, dest: int) -> Optional[int]:
        """Return the best next-hop according to current Q-table."""
        if node == dest:
            return dest
        if dest not in self.Q.get(node, {}):
            return None
        table = self.Q[node][dest]
        if not table:
            return None
        return min(table, key=table.get)

    # ── update ──────────────────────────────────────────────────────

    def update(
        self,
        node:     int,
        dest:     int,
        via:      int,
        q_delay:  float,       # queue delay at current node (measured/estimated)
        min_q_next: float,     # min Q at next node (forwarded in packet header)
    ):
        """
        Bellman update at node `node`:
            Q[node][dest][via] ← (1-η)·Q + η·(q_delay + min_Q_next)
        """
        if dest not in self.Q.get(node, {}):
            return
        if via not in self.Q[node][dest]:
            return
        old_q = self.Q[node][dest][via]
        target = q_delay + min_q_next
        self.Q[node][dest][via] += self.lr * (target - old_q)

    def min_q(self, node: int, dest: int) -> float:
        """Return min Q-value at node for dest (sent in packet header)."""
        if dest == node:
            return 0.0
        if dest not in self.Q.get(node, {}):
            return 999.0
        vals = list(self.Q[node][dest].values())
        return min(vals) if vals else 999.0

    # ── simulate routing an episode ─────────────────────────────────

    def route_episode(
        self,
        env,
        source: int,
        dest:   int,
        max_hops: int = 30,
    ) -> Tuple[List[int], float, bool]:
        """
        Simulate routing a packet from source to dest, updating Q-tables.
        Returns (path, total_delay, reached_dest).
        """
        path    = [source]
        visited = {source}
        node    = source
        total_delay = 0.0

        for _ in range(max_hops):
            if node == dest:
                break

            # Queue delay at current node (proportional to congestion)
            # Mimics M/M/1: delay ≈ congestion / (1 - congestion) × service_time
            max_congestion = 0.0
            for nb in env.G.neighbors(node):
                e = env.G.edges[node, nb]
                if e["active"]:
                    max_congestion = max(max_congestion, e["congestion"])
            q_delay = max_congestion / max(1.0 - max_congestion, 0.05) * 2.0

            next_hop = self.select_next_hop(node, dest)
            if next_hop is None:
                break

            edge = env.G.edges[node, next_hop] if env.G.has_edge(node, next_hop) else None
            if edge is None or not edge["active"]:
                # Link down — penalise and re-select
                if dest in self.Q.get(node, {}) and next_hop in self.Q[node][dest]:
                    self.Q[node][dest][next_hop] = 999.0
                break

            link_delay = edge["base_delay"] * (1.0 + 3.0 * edge["congestion"])
            total_delay += q_delay + link_delay

            # Q-routing update (packet carries min-Q from next node)
            min_q_next = self.min_q(next_hop, dest)
            self.update(node, dest, next_hop, q_delay, min_q_next)

            if next_hop in visited:
                break    # loop detected — bail out
            visited.add(next_hop)
            path.append(next_hop)
            node = next_hop

        reached = node == dest
        return path, total_delay, reached

    # ── topology update ─────────────────────────────────────────────

    def notify_link_failure(self, u: int, v: int):
        """Set Q-values to infinity for failed links (immediate reaction)."""
        for node, dest_table in self.Q.items():
            for dest, table in dest_table.items():
                if v in table and node == u:
                    table[v] = 999.0
                if u in table and node == v:
                    table[u] = 999.0

    def reset_tables(self, init_val: float = 10.0):
        for k in self.Q:
            for d in self.Q[k]:
                for n in self.Q[k][d]:
                    self.Q[k][d][n] = init_val

    def save(self, path: str):
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Convert int keys to strings for JSON serialisation
        serialisable = {
            str(k): {str(d): {str(n): v for n, v in nb.items()}
                     for d, nb in dst.items()}
            for k, dst in self.Q.items()
        }
        with open(path, "w") as f:
            json.dump(serialisable, f)

    def load(self, path: str):
        import json
        with open(path) as f:
            raw = json.load(f)
        self.Q = {
            int(k): {int(d): {int(n): v for n, v in nb.items()}
                     for d, nb in dst.items()}
            for k, dst in raw.items()
        }
