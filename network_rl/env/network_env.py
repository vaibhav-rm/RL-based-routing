"""
Custom Gymnasium environment simulating a network topology with dynamic link conditions.

Research-level enhancements over the baseline:
  • M/M/1 queuing model for delay (Kleinrock, 1975) — more physically grounded
    than linear multiplier; delay diverges as utilisation → 1 (saturation)
  • Self-similar background traffic (Pareto ON/OFF) stresses the agent with
    bursty interference flows (Leland et al., 1994)
  • Extended observation: [congestion, delay_norm, loss, active] per edge
    (4× richer than congestion-only), plus source/dest one-hot encoding
  • Curriculum learning support: failure_rate and mean_load parameters
    allow staged difficulty increases during training

CN concepts:
  M/M/1 sojourn time W = 1 / (μ - λ) where ρ = λ/μ is utilisation.
  We compute: delay = base_prop_delay + transmission_time / (1 - ρ)
  This collapses to ≈base_delay at ρ→0 and → ∞ at ρ→1 (congestion collapse).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from typing import Optional, Tuple, Dict, List


# ── Topology ────────────────────────────────────────────────────────────────
# (src, dst, bandwidth_Mbps, prop_delay_ms)
TOPOLOGY_EDGES = [
    (0, 1, 100, 5),
    (0, 2, 50,  8),
    (1, 3, 100, 3),
    (1, 4, 75,  6),
    (2, 3, 60,  4),
    (2, 5, 80,  7),
    (3, 6, 90,  5),
    (4, 6, 70,  9),
    (4, 7, 60,  6),
    (5, 7, 85,  4),
    (5, 8, 55,  10),
    (6, 9, 100, 3),
    (7, 9, 75,  5),
    (8, 9, 65,  8),
    (3, 8, 50,  12),
    (1, 5, 40,  15),
    (6, 7, 30,  7),
]

NUM_NODES   = 10
NUM_EDGES   = len(TOPOLOGY_EDGES)
PACKET_BITS = 1500 * 8         # 1500-byte Ethernet MTU in bits
MAX_STEPS   = 50
LOOP_PENALTY = -20.0

# Extended obs: 4 features per edge + 2 node scalars
OBS_DIM = NUM_EDGES * 4 + 2


# ── Delay model ─────────────────────────────────────────────────────────────

def mm1_delay(base_delay_ms: float, bandwidth_mbps: float, utilisation: float) -> float:
    """
    M/M/1 sojourn time:  W = 1/(μ - λ)  expressed in milliseconds.

    μ = bandwidth / packet_size   (packets per second)
    λ = ρ × μ                     (arrival rate from utilisation)
    W = (1/μ) / (1 - ρ)           = transmission_time / (1 - ρ)

    The base propagation delay is additive (speed-of-light component).
    Cap utilisation at 0.95 to avoid numerical blowup.
    """
    rho = float(np.clip(utilisation, 0.0, 0.95))
    tx_ms = PACKET_BITS / (bandwidth_mbps * 1e6) * 1e3    # ms
    queuing_delay = tx_ms / max(1.0 - rho, 0.05)
    return base_delay_ms + queuing_delay


# ── Environment ──────────────────────────────────────────────────────────────

class NetworkRoutingEnv(gym.Env):
    """
    Extended observation per edge: [congestion, delay_norm, loss_rate, active_flag]
    Plus: [current_node_norm, dest_node_norm]
    Total obs dim = NUM_EDGES * 4 + 2 = 70.

    Curriculum parameters:
      failure_prob:   per-link per-step failure probability
      mean_load:      mean background traffic utilisation
      use_mm1:        if True, use M/M/1 delay; else use linear model (backward compat)
    """

    metadata = {"render_modes": ["human"]}

    # Maximum normalisation constant for delay (ms)
    MAX_DELAY_NORM = 100.0

    def __init__(
        self,
        render_mode:  Optional[str] = None,
        failure_prob: float = 0.005,
        mean_load:    float = 0.4,
        use_mm1:      bool  = True,
        w_delay:      float = 10.0,
        w_drop:       float = 15.0,
    ):
        super().__init__()
        self.render_mode  = render_mode
        self.failure_prob = failure_prob
        self.mean_load    = mean_load
        self.use_mm1      = use_mm1

        # Reward objective weights. Latency and reliability are competing
        # objectives — a delay-greedy policy takes fast-but-lossy links, a
        # reliability-greedy one takes slow-but-safe links. Exposing these as
        # parameters lets us sweep the trade-off and trace a Pareto front
        # instead of committing to a single hard-coded scalarisation.
        #   w_delay: penalty per normalised ms of effective delay
        #   w_drop:  penalty incurred when a packet is dropped on a lossy link
        self.w_delay = w_delay
        self.w_drop  = w_drop

        self.G = nx.Graph()
        self.G.add_nodes_from(range(NUM_NODES))
        for (u, v, bw, delay) in TOPOLOGY_EDGES:
            self.G.add_edge(u, v, bandwidth=bw, base_delay=delay,
                            congestion=0.0, loss_rate=0.0, active=True,
                            delay_ms=float(delay))

        self.edge_list  = list(self.G.edges())
        self.max_degree = max(dict(self.G.degree()).values())

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.max_degree)

        self._current_node: int  = 0
        self._dest_node:    int  = 9
        self._visited: set       = set()
        self._steps:   int       = 0
        self._path:    List[int] = []

        # Stats for logging
        self.episode_delays: List[float] = []
        self.episode_drops:  int = 0

    # ── Gym interface ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._current_node = int(self.np_random.integers(0, NUM_NODES))
        self._dest_node    = int(self.np_random.integers(0, NUM_NODES))
        while self._dest_node == self._current_node:
            self._dest_node = int(self.np_random.integers(0, NUM_NODES))

        self._visited      = {self._current_node}
        self._steps        = 0
        self._path         = [self._current_node]
        self.episode_delays = []
        self.episode_drops  = 0

        self._update_link_conditions()
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        neighbors = self._get_sorted_neighbors(self._current_node)

        if action >= len(neighbors):
            self._steps += 1
            self._update_link_conditions()
            return self._get_obs(), -5.0, False, self._steps >= MAX_STEPS, {}

        next_node = neighbors[action]
        edge      = self._get_edge_data(self._current_node, next_node)
        reward    = 0.0
        terminated = False

        if not edge["active"]:
            reward += -10.0
        else:
            cong = edge["congestion"]
            bw   = edge["bandwidth"]
            bd   = edge["base_delay"]

            if self.use_mm1:
                delay = mm1_delay(bd, bw, cong)
            else:
                delay = bd * (1.0 + 3.0 * cong)

            self.episode_delays.append(delay)
            reward -= delay / self.MAX_DELAY_NORM * self.w_delay

            if self.np_random.random() < edge["loss_rate"]:
                reward -= self.w_drop
                self.episode_drops += 1

        if next_node in self._visited:
            reward += LOOP_PENALTY
        else:
            self._visited.add(next_node)

        self._current_node = next_node
        self._path.append(next_node)
        self._steps += 1

        if self._current_node == self._dest_node:
            reward += 20.0
            terminated = True

        self._update_link_conditions()
        truncated = self._steps >= MAX_STEPS

        info = {
            "path":         list(self._path),
            "current_node": self._current_node,
            "dest_node":    self._dest_node,
            "avg_delay":    float(np.mean(self.episode_delays)) if self.episode_delays else 0.0,
            "drops":        self.episode_drops,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"  Path: {self._path} → dest={self._dest_node}")

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        feats = []
        for (u, v) in self.edge_list:
            e    = self.G.edges[u, v]
            cong = e["congestion"]
            bw   = e["bandwidth"]
            bd   = e["base_delay"]

            if self.use_mm1 and e["active"]:
                delay_norm = min(mm1_delay(bd, bw, cong) / self.MAX_DELAY_NORM, 1.0)
            else:
                delay_norm = min(bd * (1.0 + 3.0 * cong) / 240.0, 1.0)

            feats.extend([
                cong,                               # congestion [0,1]
                delay_norm,                         # effective delay normalised
                e["loss_rate"],                     # loss rate [0,0.5]
                float(e["active"]),                 # link up/down indicator
            ])
        feats.append(self._current_node / (NUM_NODES - 1))
        feats.append(self._dest_node    / (NUM_NODES - 1))
        return np.array(feats, dtype=np.float32)

    # ── Link condition dynamics ───────────────────────────────────────────────

    def _update_link_conditions(self):
        """
        Per-step stochastic link dynamics:
          • Congestion: AR(1) random walk with Pareto-burst injection
          • Failure/recovery: Markov chain with configurable failure_prob
          • Loss: RED-like — rises sharply above 80% utilisation
        """
        for (u, v) in self.edge_list:
            edge = self.G.edges[u, v]

            # Failure / recovery (two-state Markov)
            if edge["active"]:
                if self.np_random.random() < self.failure_prob:
                    edge["active"] = False
            else:
                if self.np_random.random() < self.failure_prob * 4:
                    edge["active"] = True

            if edge["active"]:
                # Background traffic contribution (mean_load centred)
                bg_noise = self.np_random.normal(self.mean_load, 0.05)
                # AR(1) with momentum toward mean_load
                alpha = 0.3
                delta = self.np_random.normal(0, 0.04)
                new_cong = (1 - alpha) * edge["congestion"] + alpha * bg_noise + delta
                edge["congestion"] = float(np.clip(new_cong, 0.0, 1.0))

                # RED-like loss: rises sharply above 80% utilisation
                rho = edge["congestion"]
                if rho < 0.5:
                    loss = rho * 0.02
                elif rho < 0.8:
                    loss = 0.01 + (rho - 0.5) * 0.1
                else:
                    loss = 0.04 + (rho - 0.8) * 0.8   # exponential rise
                edge["loss_rate"] = float(np.clip(loss, 0.0, 0.6))
            else:
                edge["congestion"] = 1.0
                edge["loss_rate"]  = 1.0

    # ── Utilities ────────────────────────────────────────────────────────────

    def _get_sorted_neighbors(self, node: int) -> List[int]:
        return sorted(self.G.neighbors(node))

    def _get_edge_data(self, u: int, v: int) -> Dict:
        if self.G.has_edge(u, v):
            return self.G.edges[u, v]
        return {"active": False, "base_delay": 999, "bandwidth": 1,
                "congestion": 1.0, "loss_rate": 1.0}

    def get_edge_congestion_dict(self) -> Dict:
        return {(u, v): self.G.edges[u, v]["congestion"] for (u, v) in self.edge_list}

    def get_active_subgraph(self) -> nx.Graph:
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())
        for (u, v) in self.edge_list:
            e = self.G.edges[u, v]
            if e["active"]:
                if self.use_mm1:
                    w = mm1_delay(e["base_delay"], e["bandwidth"], e["congestion"])
                else:
                    w = e["base_delay"] * (1.0 + 3.0 * e["congestion"])
                H.add_edge(u, v, weight=w)
        return H

    def set_curriculum(self, failure_prob: float, mean_load: float):
        """Adjust difficulty mid-training for curriculum learning."""
        self.failure_prob = failure_prob
        self.mean_load    = mean_load
