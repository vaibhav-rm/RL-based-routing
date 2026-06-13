"""
Rainbow DQN agent — combines four improvements from the Rainbow paper
(Hessel et al., 2018):
  1. Double DQN         — decouples action selection from evaluation
  2. Dueling Networks   — separate Value + Advantage streams
  3. Prioritized Replay — sample high-TD-error transitions more often
  4. Multi-step Returns — n-step bootstrapped targets for faster credit assignment

Not included (to stay CPU-trainable in < 10 min):
  - Distributional RL (C51) — adds complexity without clear gain here
  - Noisy Networks — ε-greedy works well enough for this problem size

CN context:
  Dueling: V(s) captures global network health; A(s,a) captures
  per-link advantage. Useful when many links are equally bad.
  n-step returns: propagates reward signal faster through long paths,
  mirroring how OSPF propagates link-state updates hop-by-hop.
"""

import random
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .per_buffer import PrioritizedReplayBuffer


# ──────────────────────────────────────────────
# Dueling Q-Network
# ──────────────────────────────────────────────

class DuelingQNetwork(nn.Module):
    """
    Q(s,a) = V(s) + ( A(s,a) − mean_a A(s,a) )

    Subtracting the mean advantage makes Q identifiable
    (Wang et al., 2016, "Dueling Network Architectures").
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value_head(h)           # [B, 1]
        a = self.adv_head(h)             # [B, A]
        return v + (a - a.mean(dim=-1, keepdim=True))


# ──────────────────────────────────────────────
# N-step return buffer
# ──────────────────────────────────────────────

class NStepBuffer:
    """
    Accumulates n transitions and emits (s0, a0, G_n, s_n, done_n) where
    G_n = r0 + γ·r1 + … + γ^(n-1)·r_{n-1}.
    This lengthens the effective horizon so the agent sees consequences
    of its routing decisions n hops ahead.
    """

    def __init__(self, n: int, gamma: float):
        self.n     = n
        self.gamma = gamma
        self.buf: deque = deque(maxlen=n)

    def push(self, state, action, reward, next_state, done) -> Optional[tuple]:
        self.buf.append((state, action, reward, next_state, done))
        if len(self.buf) < self.n:
            return None
        # Compute n-step discounted return
        G = 0.0
        for i, (s, a, r, ns, d) in enumerate(self.buf):
            G += (self.gamma ** i) * r
            if d:
                break
        s0, a0, _, _,  _  = self.buf[0]
        sn, _, _, ns_last, d_last = self.buf[-1]
        return (s0, a0, G, ns_last, d_last)

    def flush(self):
        """Drain remaining transitions at episode end."""
        results = []
        while len(self.buf) > 0:
            G = 0.0
            for i, (s, a, r, ns, d) in enumerate(self.buf):
                G += (self.gamma ** i) * r
                if d:
                    break
            s0, a0, _, _, _ = self.buf[0]
            _, _, _, ns_last, d_last = self.buf[-1]
            results.append((s0, a0, G, ns_last, d_last))
            self.buf.popleft()
        return results

    def reset(self):
        self.buf.clear()


# ──────────────────────────────────────────────
# Rainbow Agent
# ──────────────────────────────────────────────

class RainbowAgent:

    def __init__(
        self,
        state_dim:      int,
        action_dim:     int,
        lr:             float = 5e-4,
        gamma:          float = 0.95,
        n_step:         int   = 3,
        epsilon:        float = 1.0,
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.995,
        batch_size:     int   = 64,
        target_update:  int   = 10,
        hidden:         int   = 256,
        per_alpha:      float = 0.6,
        per_beta_start: float = 0.4,
        buffer_cap:     int   = 100_000,
        device:         str   = "cpu",
    ):
        self.state_dim    = state_dim
        self.action_dim   = action_dim
        self.gamma        = gamma
        self.n_step       = n_step
        self.epsilon      = epsilon
        self.epsilon_min  = epsilon_min
        self.epsilon_decay= epsilon_decay
        self.batch_size   = batch_size
        self.target_update= target_update
        self.device       = torch.device(device)

        self.policy_net = DuelingQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, hidden).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay     = PrioritizedReplayBuffer(
            capacity=buffer_cap, alpha=per_alpha, beta_start=per_beta_start,
            beta_frames=buffer_cap * 2,
        )
        self.nstep_buf  = NStepBuffer(n_step, gamma)
        self.steps      = 0
        self.episode    = 0

    # ── action selection ────────────────────────────────────────────

    def select_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        # Warmup is training-only; a loaded (eval-mode) net is used directly so
        # evaluation reflects the trained policy rather than random routing.
        if random.random() < self.epsilon or \
                (self.policy_net.training and len(self.replay) < self.batch_size):
            return random.choice(valid_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t).squeeze(0).cpu().numpy()
        masked = np.full(self.action_dim, -np.inf)
        for a in valid_actions:
            masked[a] = q[a]
        return int(np.argmax(masked))

    # ── experience storage ──────────────────────────────────────────

    def store(self, state, action, reward, next_state, done):
        transition = self.nstep_buf.push(state, action, reward, next_state, done)
        if transition is not None:
            self.replay.push(*transition)
        if done:
            for t in self.nstep_buf.flush():
                self.replay.push(*t)
            self.nstep_buf.reset()

    # ── learning ────────────────────────────────────────────────────

    def learn(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay.sample(self.batch_size)

        st  = torch.FloatTensor(states).to(self.device)
        at  = torch.LongTensor(actions).to(self.device)
        rt  = torch.FloatTensor(rewards).to(self.device)
        nst = torch.FloatTensor(next_states).to(self.device)
        dt  = torch.FloatTensor(dones).to(self.device)
        wt  = torch.FloatTensor(weights).to(self.device)

        q_current = self.policy_net(st).gather(1, at.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: policy net selects, target net evaluates
            next_actions = self.policy_net(nst).argmax(1)
            q_next = self.target_net(nst).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # n-step discount
            q_target = rt + (self.gamma ** self.n_step) * q_next * (1.0 - dt)

        td_errors = (q_target - q_current).detach().cpu().numpy()
        # Weighted MSE loss (IS correction)
        loss = (wt * (q_current - q_target.detach()) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.replay.update_priorities(indices, np.abs(td_errors) + 1e-6)
        self.steps += 1
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode += 1

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "steps":   self.steps,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ck["policy"])
        self.target_net.load_state_dict(ck["target"])
        self.epsilon = ck.get("epsilon", self.epsilon_min)
        self.steps   = ck.get("steps", 0)
        self.policy_net.eval()
        self.target_net.eval()
