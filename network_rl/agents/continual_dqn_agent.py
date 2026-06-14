"""
Continual-learning DQN agent — Elastic Weight Consolidation (EWC) for routing.

Catastrophic forgetting: when a single network is trained on task B after task A,
gradient updates for B overwrite the weights that encoded A, and performance on A
collapses. This is a recognised obstacle for routing agents that must adapt to a
new traffic/failure/destination regime without forgetting how to serve the old
one — yet continual-RL remedies (well studied in e.g. anti-jamming) are largely
absent from the routing literature.

EWC (Kirkpatrick et al., PNAS 2017) mitigates this by anchoring the weights that
mattered most for previous tasks. After finishing a task we estimate each
parameter's importance via the diagonal of the Fisher information, snapshot the
converged weights θ*, and on later tasks add a quadratic penalty

    L_EWC = (λ/2) · Σ_i F_i · (θ_i − θ*_i)²

that resists moving important weights while leaving unimportant ones free to
learn the new task. The penalty is injected through DQNAgent._extra_loss, so the
base TD update is untouched.
"""

import torch

from network_rl.agents.dqn_agent import DQNAgent


class ContinualDQNAgent(DQNAgent):
    """DQNAgent + EWC: consolidate() after each task, penalty applied thereafter."""

    def __init__(self, *args, ewc_lambda: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        # One (star_params, fisher) entry per consolidated task.
        self._ewc_tasks: list = []

    # ── EWC machinery ────────────────────────────────────────────────────────

    def estimate_fisher(self, n_batches: int = 32) -> dict:
        """
        Diagonal Fisher information over the current replay buffer (the just-
        finished task's experience): the mean squared gradient of the TD loss
        w.r.t. each parameter. Larger ⇒ the parameter mattered more for this task.
        """
        fisher = {n: torch.zeros_like(p) for n, p in self.policy_net.named_parameters()}
        if len(self.replay) < self.batch_size:
            return fisher

        n = 0
        for _ in range(n_batches):
            states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
            states_t  = torch.FloatTensor(states).to(self.device)
            actions_t = torch.LongTensor(actions).to(self.device)
            rewards_t = torch.FloatTensor(rewards).to(self.device)
            next_t    = torch.FloatTensor(next_states).to(self.device)
            dones_t   = torch.FloatTensor(dones).to(self.device)

            q_cur = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                na = self.policy_net(next_t).argmax(1)
                q_next = self.target_net(next_t).gather(1, na.unsqueeze(1)).squeeze(1)
                q_tgt = rewards_t + self.gamma * q_next * (1.0 - dones_t)
            loss = torch.nn.functional.mse_loss(q_cur, q_tgt)

            self.optimizer.zero_grad()
            loss.backward()
            for name, p in self.policy_net.named_parameters():
                if p.grad is not None:
                    fisher[name] += p.grad.detach() ** 2
            n += 1

        if n > 0:
            for name in fisher:
                fisher[name] /= n
        self.optimizer.zero_grad()

        # Normalise so the MEAN importance is 1. Raw Fisher from a converged
        # buffer is tiny and varies by orders of magnitude across layers, which
        # made the penalty negligible at any λ; normalising preserves the
        # *relative* importance ranking while putting λ on a meaningful, scale-
        # stable footing comparable to the TD loss.
        total = sum(float(f.sum()) for f in fisher.values())
        count = sum(f.numel() for f in fisher.values())
        mean_f = total / max(count, 1)
        if mean_f > 0:
            for name in fisher:
                fisher[name] = fisher[name] / mean_f
        return fisher

    def consolidate(self):
        """Snapshot current weights + their Fisher importance as a protected task."""
        star = {n: p.detach().clone() for n, p in self.policy_net.named_parameters()}
        self._ewc_tasks.append((star, self.estimate_fisher()))

    def _extra_loss(self):
        if self.ewc_lambda == 0.0 or not self._ewc_tasks:
            return 0.0
        penalty = 0.0
        params = dict(self.policy_net.named_parameters())
        for star, fisher in self._ewc_tasks:
            for name, p in params.items():
                penalty = penalty + (fisher[name] * (p - star[name]) ** 2).sum()
        return 0.5 * self.ewc_lambda * penalty
