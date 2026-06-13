"""
Real-Network → DQN State Adapter.

CN concepts:
- Bridges the gap between physical measurements (RTT, loss%) and the
  normalised [0,1] congestion features the DQN was trained on.
- RTT is mapped to congestion using a saturation model: at max_rtt_ms the
  link is considered fully saturated (congestion=1), mirroring how TCP
  Cubic/BBR reduce cwnd when RTT increases beyond baseline.
- Loss% maps linearly to the loss_rate slot in the simulated env state.
- Unknown links (not yet probed) default to congestion=0.5 (neutral prior),
  consistent with OSPF's initial cost assumption before LSAs arrive.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# These must match constants in network_env.py
from network_rl.env.network_env import NUM_EDGES, NUM_NODES

# Calibration: RTT above this → congestion = 1.0  (adjust per your LAN)
MAX_RTT_MS   = 200.0
# Loss above this % → loss_rate = 1.0
MAX_LOSS_PCT = 50.0


class RealEnvAdapter:
    """
    Maintains a mapping from (node_a, node_b) → measured metrics and
    produces the same state vector shape the DQN was trained on.

    node_map: dict mapping simulated node IDs (0-9) to (ip, port) tuples.
    edge_order: list of (u, v) tuples in the same order as NetworkRoutingEnv.edge_list.
    """

    def __init__(
        self,
        node_map: Dict[int, Tuple[str, int]],
        edge_order: List[Tuple[int, int]],
        port: int = 9999,
    ):
        self.node_map   = node_map      # {sim_id: (ip, port)}
        self.edge_order = edge_order    # must match env.edge_list ordering
        self.port       = port
        # Latest probe results keyed by frozenset({u, v})
        self._metrics: Dict[frozenset, Dict] = {}

    # ------------------------------------------------------------------

    def set_link_metrics(self, node_a: int, node_b: int, rtt_ms: float, loss_pct: float):
        """Directly set metrics for a (node_a, node_b) pair."""
        key = frozenset({node_a, node_b})
        self._metrics[key] = {"rtt_ms": rtt_ms, "loss_pct": loss_pct}

    # ------------------------------------------------------------------

    def _edge_congestion(self, u: int, v: int) -> float:
        key = frozenset({u, v})
        if key not in self._metrics:
            return 0.5  # neutral prior (link not yet measured)
        m = self._metrics[key]
        rtt  = m.get("rtt_ms",   -1)
        loss = m.get("loss_pct",  0)
        if rtt < 0:
            return 1.0  # unreachable → fully congested
        cong_rtt  = min(rtt / MAX_RTT_MS, 1.0)
        cong_loss = min(loss / MAX_LOSS_PCT, 1.0)
        # Weighted combination: delay contributes more than loss
        return float(np.clip(0.7 * cong_rtt + 0.3 * cong_loss, 0.0, 1.0))

    # Must match NetworkRoutingEnv.MAX_DELAY_NORM so measured delay lands on the
    # same scale the agent was trained on.
    MAX_DELAY_NORM = 100.0

    def _edge_features(self, u: int, v: int) -> List[float]:
        """
        The 4 per-edge features the agent was trained on, derived from real probes:
          [congestion, delay_norm, loss_rate, active]
        Mirrors NetworkRoutingEnv._get_obs so the observation is interchangeable.
        """
        key  = frozenset({u, v})
        m    = self._metrics.get(key)
        cong = self._edge_congestion(u, v)
        if m is None:
            # Link not yet probed → neutral prior, assumed up.
            return [cong, 0.5, 0.0, 1.0]
        rtt  = m.get("rtt_ms",  -1)
        loss = m.get("loss_pct", 0.0)
        active     = 1.0 if rtt >= 0 else 0.0
        delay_norm = min(rtt / self.MAX_DELAY_NORM, 1.0) if rtt >= 0 else 1.0
        loss_rate  = float(np.clip(loss / 100.0, 0.0, 0.5))
        return [cong, delay_norm, loss_rate, active]

    def build_state(self, current_node: int, dest_node: int) -> np.ndarray:
        """
        Build the agent input vector in the same layout as NetworkRoutingEnv._get_obs:
          [congestion, delay_norm, loss_rate, active] × NUM_EDGES,
          then [current_node_norm, dest_node_norm].
        Total length = NUM_EDGES * 4 + 2 (= OBS_DIM).
        """
        feats: List[float] = []
        for (u, v) in self.edge_order:
            feats.extend(self._edge_features(u, v))
        feats.append(current_node / (NUM_NODES - 1))
        feats.append(dest_node    / (NUM_NODES - 1))
        return np.array(feats, dtype=np.float32)

    def get_neighbors(self, node: int, graph) -> List[int]:
        """Return sorted neighbours of node in the (possibly partial) graph."""
        return sorted(graph.neighbors(node))

    def congestion_summary(self) -> str:
        """Human-readable table of current link congestion estimates."""
        lines = ["Edge       Congestion  RTT(ms)  Loss%"]
        lines.append("-" * 38)
        for u, v in self.edge_order:
            key = frozenset({u, v})
            if key in self._metrics:
                m = self._metrics[key]
                c = self._edge_congestion(u, v)
                lines.append(f"  {u:2d}–{v:<2d}     {c:.2f}      "
                              f"{m.get('rtt_ms', -1):6.1f}  {m.get('loss_pct', 0):5.1f}")
            else:
                lines.append(f"  {u:2d}–{v:<2d}     0.50 (no data)")
        return "\n".join(lines)
