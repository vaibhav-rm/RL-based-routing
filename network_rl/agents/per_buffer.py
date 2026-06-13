"""
Prioritized Experience Replay (PER) — Schaul et al. 2016
"Prioritized Experience Replay", ICLR 2016.

Key idea: sample transitions with probability proportional to |TD-error|^α.
Correct the resulting bias with Importance Sampling (IS) weights β → 1.

Sum-tree gives O(log N) for both insert and sample.
"""

import numpy as np
from typing import List, Tuple


# ──────────────────────────────────────────────
# Sum-tree
# ──────────────────────────────────────────────

class SumTree:
    """
    Binary tree where each leaf stores a priority p_i and each
    internal node stores the sum of its children.
    Leaf range: [capacity-1, 2*capacity-2] in the flat array.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = [None] * capacity
        self.write    = 0          # circular write pointer
        self.size     = 0

    # ── write ──────────────────────────────────

    def add(self, priority: float, data):
        leaf = self.write + self.capacity - 1
        self.data[self.write] = data
        self._update(leaf, priority)
        self.write = (self.write + 1) % self.capacity
        self.size  = min(self.size + 1, self.capacity)

    def _update(self, leaf_idx: int, priority: float):
        delta = priority - self.tree[leaf_idx]
        self.tree[leaf_idx] = priority
        idx = leaf_idx
        while idx > 0:
            idx = (idx - 1) >> 1      # parent
            self.tree[idx] += delta

    def update(self, leaf_idx: int, priority: float):
        self._update(leaf_idx, priority)

    # ── read ───────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def get(self, s: float) -> Tuple[int, float, object]:
        """Descend tree to find leaf whose prefix sum ≥ s."""
        idx = 0
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


# ──────────────────────────────────────────────
# Prioritized Replay Buffer
# ──────────────────────────────────────────────

class PrioritizedReplayBuffer:
    """
    Implements Algorithm 1 from Schaul et al. 2016.

    alpha:       exponent for priority → prob conversion (0=uniform, 1=greedy)
    beta_start:  initial IS correction exponent (annealed to 1 over training)
    beta_frames: number of steps to anneal beta to 1.0
    """

    def __init__(
        self,
        capacity:     int   = 100_000,
        alpha:        float = 0.6,
        beta_start:   float = 0.4,
        beta_frames:  int   = 200_000,
        epsilon:      float = 1e-6,
    ):
        self.tree         = SumTree(capacity)
        self.alpha        = alpha
        self.beta         = beta_start
        self.beta_inc     = (1.0 - beta_start) / beta_frames
        self.max_priority = 1.0
        self.epsilon      = epsilon

    # ── public API ─────────────────────────────

    def push(self, *transition):
        """New transitions get max priority so they're sampled at least once."""
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size: int):
        """
        Returns (states, actions, rewards, next_states, dones,
                 tree_indices, is_weights).
        tree_indices are needed to call update_priorities() after learning.
        """
        self.beta = min(1.0, self.beta + self.beta_inc)

        segment = self.tree.total / batch_size
        indices, priorities, batch_data = [], [], []

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, pri, data = self.tree.get(s)
            if data is None:
                # Buffer not yet full — resample uniformly from non-None
                for j in np.random.permutation(self.tree.capacity):
                    if self.tree.data[j] is not None:
                        idx  = j + self.tree.capacity - 1
                        pri  = self.tree.tree[idx]
                        data = self.tree.data[j]
                        break
            indices.append(idx)
            priorities.append(pri)
            batch_data.append(data)

        # IS weights
        probs   = np.array(priorities, dtype=np.float64) / self.tree.total
        probs   = np.clip(probs, 1e-12, 1.0)
        weights = (self.tree.size * probs) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch_data)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.size
