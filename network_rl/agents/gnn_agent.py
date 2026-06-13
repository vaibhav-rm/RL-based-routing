"""
Graph Neural Network (GNN) routing agent — vectorised implementation.

Research contribution:
  Standard MLP-DQN requires retraining for every new topology.
  A GNN encodes topology structure via message passing, enabling
  zero-shot generalisation to unseen graphs — a key open problem
  in learned network control (see: Rusek et al. RouteNet, 2020).

Architecture:
  GraphSAGE-style (Hamilton et al., 2017) with vectorised scatter operations.
  • Node features: [degree_norm, is_current, is_dest, avg_cong]
  • Edge features: [congestion, delay_norm, loss_rate, active]
  • 2 message-passing layers → per-node embeddings
  • Q-head: MLP( cat(emb_v, emb_n) ) for each neighbour n

Performance:
  Vectorised with index_add_ — O(E) per forward pass, not O(N×degree).
  Suitable for training 800 episodes in <3 minutes on CPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from typing import List, Tuple

import networkx as nx

NODE_FEAT_DIM = 4
EDGE_FEAT_DIM = 4
MAX_DELAY_MS  = 15.0


# ── Vectorised SAGEConv ────────────────────────────────────────────────────

class SAGEConv(nn.Module):
    """
    GraphSAGE mean-aggregator with scatter-based vectorisation.
    h_v' = ReLU( W_self·h_v  +  W_nb·mean_{u∈N(v)} cat(h_u, e_uv) + b )
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim,          out_dim, bias=False)
        self.W_nb   = nn.Linear(in_dim + edge_dim, out_dim, bias=False)
        self.bias   = nn.Parameter(torch.zeros(out_dim))
        self.out_dim = out_dim

    def forward(
        self,
        h:            torch.Tensor,    # [N, in_dim]
        edge_index:   torch.Tensor,    # [2, E]  (src, dst)
        edge_feats:   torch.Tensor,    # [E, edge_dim]
        N:            int,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_src   = h[src]                                    # [E, in_dim]
        msgs    = torch.cat([h_src, edge_feats], dim=-1)   # [E, in+edge]
        msgs_t  = self.W_nb(msgs)                           # [E, out_dim]

        # Scatter-mean: aggregate messages to destination nodes
        agg   = torch.zeros(N, self.out_dim, device=h.device)
        count = torch.zeros(N, 1,            device=h.device)
        ones  = torch.ones(src.shape[0], 1,  device=h.device)
        agg.index_add_(0, dst, msgs_t)
        count.index_add_(0, dst, ones)
        agg = agg / count.clamp(min=1.0)

        return F.relu(self.W_self(h) + agg + self.bias)


# ── GNN Q-Network ──────────────────────────────────────────────────────────

class GNNQNetwork(nn.Module):
    """
    Two SAGEConv layers → Q-value for each (current_node, neighbour) pair.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.hidden = hidden
        self.conv1  = SAGEConv(NODE_FEAT_DIM, EDGE_FEAT_DIM, hidden)
        self.conv2  = SAGEConv(hidden,        EDGE_FEAT_DIM, hidden)
        self.q_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_feats:  torch.Tensor,   # [N, 4]
        edge_index:  torch.Tensor,   # [2, E]
        edge_feats:  torch.Tensor,   # [E, 4]
        current:     int,
        neighbors:   List[int],
        N:           int,
    ) -> torch.Tensor:
        h1 = self.conv1(node_feats, edge_index, edge_feats, N)
        h2 = self.conv2(h1,         edge_index, edge_feats, N)
        emb_v = h2[current]
        q_vals = [self.q_head(torch.cat([emb_v, h2[nb]], dim=-1)).squeeze(-1)
                  for nb in neighbors]
        return torch.stack(q_vals) if q_vals else torch.zeros(1)


# ── Feature extraction ─────────────────────────────────────────────────────

def build_graph_tensors(
    graph:       nx.Graph,
    edge_list:   List[Tuple[int, int]],
    current:     int,
    dest:        int,
    max_deg:     int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      node_feats [N, 4], edge_index [2, 2E] (both directions), edge_feats [2E, 4]
    """
    N = graph.number_of_nodes()
    max_d = max(dict(graph.degree()).values()) or 1

    # Node features
    nf = torch.zeros(N, NODE_FEAT_DIM)
    for n in range(N):
        nbrs = list(graph.neighbors(n))
        congs = [graph.edges[n, nb]["congestion"]
                 for nb in nbrs if graph.has_edge(n, nb) and graph.edges[n, nb]["active"]]
        avg_c = float(np.mean(congs)) if congs else 0.5
        nf[n] = torch.tensor([
            graph.degree(n) / max_d,
            float(n == current),
            float(n == dest),
            avg_c,
        ])

    # Edge features (both directions for undirected graph)
    src_list, dst_list, ef_list = [], [], []
    for (u, v) in edge_list:
        e = graph.edges[u, v]
        feat = [
            e.get("congestion", 0.0),
            e.get("base_delay", 5.0) / MAX_DELAY_MS,
            e.get("loss_rate", 0.0),
            float(e.get("active", True)),
        ]
        src_list += [u, v]; dst_list += [v, u]
        ef_list  += [feat, feat]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_feats = torch.tensor(ef_list, dtype=torch.float32)
    return nf, edge_index, edge_feats


# ── GNN Agent ─────────────────────────────────────────────────────────────

class GNNAgent:

    def __init__(
        self,
        graph:         nx.Graph,
        edge_list:     List[Tuple[int, int]],
        hidden:        int   = 64,
        lr:            float = 5e-4,
        gamma:         float = 0.95,
        epsilon:       float = 1.0,
        epsilon_min:   float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size:    int   = 32,
        buffer_cap:    int   = 20_000,
        target_update: int   = 10,
        device:        str   = "cpu",
    ):
        self.graph      = graph
        self.edge_list  = edge_list
        self.N          = graph.number_of_nodes()
        self.max_degree = max(dict(graph.degree()).values())
        self.gamma      = gamma
        self.epsilon    = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device     = torch.device(device)

        self.policy_net = GNNQNetwork(hidden).to(self.device)
        self.target_net = GNNQNetwork(hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay: deque = deque(maxlen=buffer_cap)
        self.steps   = 0
        self.episode = 0

    def _tensors(self, env, current: int, dest: int):
        return build_graph_tensors(env.G, self.edge_list, current, dest, self.max_degree)

    def select_action(self, env, current: int, dest: int) -> Tuple[int, List[int]]:
        neighbors = sorted(env.G.neighbors(current))
        if not neighbors:
            return 0, neighbors
        # Warmup is training-only; a loaded (eval-mode) net is used directly so
        # evaluation reflects the trained policy rather than random routing.
        if random.random() < self.epsilon or \
                (self.policy_net.training and len(self.replay) < self.batch_size):
            return random.randrange(len(neighbors)), neighbors

        nf, ei, ef = self._tensors(env, current, dest)
        with torch.no_grad():
            q = self.policy_net(nf, ei, ef, current, neighbors, self.N)
        return int(q.argmax().item()), neighbors

    def rank_actions(self, env, current: int, dest: int) -> Tuple[List[int], List[int]]:
        """
        Return (ranked_indices, neighbors): indices into the sorted-neighbour list
        ordered by descending Q-value. Lets the real-network data plane pick the
        agent's best *reachable* next hop after masking failed/visited links.
        """
        neighbors = sorted(env.G.neighbors(current))
        if not neighbors:
            return [], neighbors
        nf, ei, ef = self._tensors(env, current, dest)
        with torch.no_grad():
            q = self.policy_net(nf, ei, ef, current, neighbors, self.N)
        order = sorted(range(len(neighbors)), key=lambda i: float(q[i]), reverse=True)
        return order, neighbors

    def store(self, s_tuple, action_idx, reward, ns_tuple, done):
        self.replay.append((s_tuple, action_idx, reward, ns_tuple, done))

    def learn(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay, self.batch_size)
        total_loss = torch.tensor(0.0)
        self.optimizer.zero_grad()

        for (s_t, a_idx, rew, ns_t, done) in batch:
            env_g, cur, dst, nbrs = s_t
            nf, ei, ef = build_graph_tensors(env_g, self.edge_list,
                                             cur, dst, self.max_degree)
            q_vals = self.policy_net(nf, ei, ef, cur, nbrs, self.N)
            if len(q_vals) == 0:
                continue
            a_idx  = min(a_idx, len(q_vals) - 1)
            q_cur  = q_vals[a_idx]

            env_g2, cur2, dst2, nbrs2 = ns_t
            nf2, ei2, ef2 = build_graph_tensors(env_g2, self.edge_list,
                                                cur2, dst2, self.max_degree)
            with torch.no_grad():
                q_next_vals = self.target_net(nf2, ei2, ef2, cur2, nbrs2, self.N)
                q_next = q_next_vals.max() if len(q_next_vals) > 0 else torch.tensor(0.0)
            q_tgt = torch.tensor(rew, dtype=torch.float32) + \
                    self.gamma * q_next * (1.0 - float(done))
            total_loss = total_loss + F.mse_loss(q_cur, q_tgt)

        if total_loss.item() > 0:
            (total_loss / self.batch_size).backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.optimizer.step()

        self.steps += 1
        return float(total_loss.item() / self.batch_size)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    def sync_target(self):
        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({"policy": self.policy_net.state_dict(),
                    "epsilon": self.epsilon, "steps": self.steps}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck["policy"])
        self.epsilon = ck.get("epsilon", self.epsilon_min)
        self.steps   = ck.get("steps", 0)
        self.policy_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())
