"""
Realistic traffic generators for the network simulation.

CN concepts:
  Poisson arrivals (M in M/M/1): memoryless inter-arrival times are the
  standard assumption in teletraffic theory (Erlang, 1909).
  Self-similar / Pareto bursts: real internet traffic is bursty with
  heavy-tailed inter-arrival times (Leland et al., 1994 — the seminal
  paper on self-similarity in LAN traffic from Bellcore).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Flow:
    """A single traffic flow with source, destination, and rate."""
    src:      int
    dst:      int
    rate_mbps: float    # mean bit rate
    priority:  int = 0  # 0=best-effort, 1=video, 2=voice (QoS class)


class PoissonTrafficModel:
    """
    Generates Poisson packet arrivals (rate = λ packets/sec).
    Each link's utilisation is λ / μ where μ = bandwidth / packet_size.
    """

    def __init__(self, mean_load: float = 0.4, dt: float = 1.0):
        """
        mean_load: target mean utilisation across all links [0,1]
        dt:        simulation timestep (seconds)
        """
        self.mean_load = mean_load
        self.dt        = dt

    def sample_load(self, bandwidth_mbps: float) -> float:
        """Return a Poisson-sampled utilisation for a link of given bandwidth."""
        # λ = mean_load × μ; add Gaussian noise for per-link variation
        noise = np.random.normal(0, 0.05)
        load  = np.clip(self.mean_load + noise, 0.0, 0.99)
        return float(load)


class SelfSimilarTrafficModel:
    """
    Pareto ON/OFF traffic model producing self-similar (bursty) load.
    ON period: heavy-tail Pareto(α=1.5); OFF period: exponential.
    Reference: Willinger et al., "Self-Similarity Through High-Variability", 1997.
    """

    def __init__(self, mean_load: float = 0.4, alpha: float = 1.5):
        self.mean_load = mean_load
        self.alpha     = alpha         # Pareto shape (1.5 → heavy tail)
        self._state:   np.ndarray = np.array([])   # per-link ON/OFF state
        self._counter: np.ndarray = np.array([])   # remaining duration

    def initialise(self, n_links: int):
        self._state   = np.random.choice([0.0, 1.0], size=n_links)
        self._counter = np.zeros(n_links)

    def step(self, n_links: int) -> np.ndarray:
        """Advance one timestep; return utilisation for each link."""
        if len(self._state) != n_links:
            self.initialise(n_links)

        loads = np.zeros(n_links)
        for i in range(n_links):
            if self._counter[i] <= 0:
                if self._state[i] == 1.0:
                    # Transitioning to OFF: exponential duration
                    self._state[i]   = 0.0
                    self._counter[i] = np.random.exponential(5.0)
                else:
                    # Transitioning to ON: Pareto duration (heavy-tailed burst)
                    self._state[i]   = 1.0
                    # Pareto sample: xm * U^(-1/alpha)
                    self._counter[i] = 1.0 * (np.random.uniform() ** (-1.0 / self.alpha))
            else:
                self._counter[i] -= 1

            if self._state[i] == 1.0:
                loads[i] = np.clip(
                    np.random.normal(self.mean_load * 1.5, 0.1), 0.0, 0.99
                )
            else:
                loads[i] = np.clip(
                    np.random.normal(self.mean_load * 0.3, 0.05), 0.0, 0.5
                )
        return loads


class TrafficMatrix:
    """
    Multi-commodity flow: N simultaneous flows with different src/dst pairs.
    Each flow contributes load to links along its path.

    Used to simulate background traffic that interferes with the routed packet,
    analogous to competing TCP flows in a real network.
    """

    def __init__(
        self,
        flows: List[Flow],
        n_nodes: int,
    ):
        self.flows   = flows
        self.n_nodes = n_nodes
        # Gravity model: load proportional to rate of each flow
        self._total_rate = sum(f.rate_mbps for f in flows) or 1.0

    @classmethod
    def random_matrix(
        cls,
        n_nodes: int,
        n_flows: int = 6,
        max_rate_mbps: float = 20.0,
    ) -> "TrafficMatrix":
        """Generate a random traffic matrix."""
        flows = []
        nodes = list(range(n_nodes))
        for _ in range(n_flows):
            src = np.random.choice(nodes)
            dst = np.random.choice([n for n in nodes if n != src])
            rate = np.random.uniform(1.0, max_rate_mbps)
            pri  = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            flows.append(Flow(src=int(src), dst=int(dst),
                              rate_mbps=float(rate), priority=int(pri)))
        return cls(flows, n_nodes)

    def edge_load(self, edge_list, graph) -> dict:
        """
        Compute approximate load contribution on each edge from background flows.
        Uses simple shortest-path routing for background flows.
        """
        import networkx as nx
        loads = {e: 0.0 for e in edge_list}
        for flow in self.flows:
            try:
                path = nx.shortest_path(graph, flow.src, flow.dst)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for u, v in zip(path[:-1], path[1:]):
                key = (u, v) if (u, v) in loads else (v, u)
                if key in loads:
                    # Add fractional load (rate / bandwidth)
                    bw = graph.edges[u, v].get("bandwidth", 100.0)
                    loads[key] += min(flow.rate_mbps / bw, 0.3)
        return loads
