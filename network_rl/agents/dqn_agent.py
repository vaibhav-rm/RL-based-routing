"""
Deep Q-Network (DQN) agent for adaptive packet routing.

CN concepts:
- The Q-table is replaced by a neural net (handles large continuous state spaces
  that arise from per-link congestion metrics, unlike Bellman-Ford's discrete tables).
- Experience replay breaks temporal correlation between consecutive link-state
  observations — analogous to how TCP pacing smooths bursty traffic.
- Target network stabilises training: without it, the "moving target" resembles
  oscillating routing metrics in a network with no dampening.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ------------------------------------------------------------------
# Neural Network
# ------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Fully-connected network: state → Q-values for each possible next-hop action.
    Three hidden layers give enough capacity to learn multi-hop path preferences.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------
# Replay Buffer
# ------------------------------------------------------------------

class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done) transitions.
    Random mini-batch sampling de-correlates temporally adjacent experiences,
    which is critical when link congestion follows autocorrelated time series.
    """

    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ------------------------------------------------------------------
# DQN Agent
# ------------------------------------------------------------------

class DQNAgent:
    """
    Double-DQN with ε-greedy exploration and periodic target-network sync.

    Hyperparameters tuned so training converges in ~500 episodes on CPU.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float          = 1e-3,
        gamma: float       = 0.95,    # discount: future hops matter less (like TTL decay)
        epsilon: float     = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int    = 64,
        target_update: int = 10,      # sync target net every N episodes
        buffer_capacity: int = 50_000,
        device: str        = "cpu",
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay
        self.batch_size   = batch_size
        self.target_update= target_update
        self.device       = torch.device(device)

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay    = ReplayBuffer(buffer_capacity)
        self.steps     = 0
        self.episode   = 0

    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, valid_actions: list) -> int:
        """
        ε-greedy selection restricted to valid_actions (available next hops).
        Masking invalid actions prevents the agent from choosing a non-existent
        neighbour — equivalent to checking the routing table before forwarding.
        """
        # Warmup (act randomly until the buffer can supply a batch) applies only
        # while training. After load()/eval the net is in eval mode and is used
        # directly — otherwise a freshly loaded agent would route randomly.
        if random.random() < self.epsilon or \
                (self.policy_net.training and len(self.replay) < self.batch_size):
            return random.choice(valid_actions)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        # Mask invalid actions with -inf
        masked = np.full(self.action_dim, -np.inf)
        for a in valid_actions:
            masked[a] = q_values[a]
        return int(np.argmax(masked))

    def rank_actions(self, state: np.ndarray, valid_actions: list) -> list:
        """
        Return `valid_actions` ordered by descending Q-value (greedy preference).

        Used by the real-network data plane to pick the agent's best *reachable*
        next hop: the controller masks failed/visited links and forwards to the
        highest-ranked survivor, so a single down link doesn't strand a packet.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t).squeeze(0).cpu().numpy()
        return sorted(valid_actions, key=lambda a: q[a], reverse=True)

    def store(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def learn(self) -> float:
        """One gradient-descent step. Returns the loss value."""
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_t     = torch.FloatTensor(states).to(self.device)
        actions_t    = torch.LongTensor(actions).to(self.device)
        rewards_t    = torch.FloatTensor(rewards).to(self.device)
        next_states_t= torch.FloatTensor(next_states).to(self.device)
        dones_t      = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_current = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double-DQN: policy net selects action, target net evaluates it)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1)
            q_next = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        loss = nn.MSELoss()(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    def sync_target(self):
        """Copy policy weights to target network (hard update)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "steps":   self.steps,
        }, path)
        print(f"[DQN] Model saved → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.steps   = checkpoint.get("steps",   0)
        self.policy_net.eval()
        self.target_net.eval()
        print(f"[DQN] Model loaded ← {path}")
