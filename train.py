"""
Multi-agent training script with curriculum learning and CSV logging.

Trains four agents in sequence (DQN, Rainbow, GNN, Q-Routing) across
multiple random seeds. Produces per-episode CSV logs compatible with
TensorBoard (via tensorboard --logdir logs/) and custom plot scripts.

Key research features:
  • Curriculum learning: staged difficulty increase during training
  • Multi-seed: 5 seeds per algorithm for statistical validity
  • CSV logging: episode reward, loss, epsilon, avg_delay per step
  • Checkpoint saving every N episodes

Usage:
    python train.py [--config experiments/config.yaml]
                    [--agents dqn rainbow gnn qrouting]
                    [--episodes 800] [--seeds 5]
"""

import os, sys, csv, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from network_rl.env.network_env  import NetworkRoutingEnv, OBS_DIM, NUM_NODES
from network_rl.agents.dqn_agent    import DQNAgent
from network_rl.agents.rainbow_agent import RainbowAgent
from network_rl.agents.gnn_agent    import GNNAgent
from network_rl.agents.q_routing    import QRoutingAgent
from experiments.config import load_config, ExperimentConfig

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
LOGS_DIR    = os.path.join(os.path.dirname(__file__), "logs")

for d in [MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def valid_actions(env):
    return list(range(len(env._get_sorted_neighbors(env._current_node))))


def apply_curriculum(env, episode: int, curriculum: list):
    """Advance curriculum stage based on episode count."""
    for thresh, fp, ml in reversed(curriculum):
        if episode >= thresh:
            env.set_curriculum(failure_prob=fp, mean_load=ml)
            return fp, ml
    return env.failure_prob, env.mean_load


class CSVLogger:
    def __init__(self, path: str):
        self.f   = open(path, "w", newline="")
        self.csv = csv.writer(self.f)
        self.csv.writerow(["episode", "reward", "loss", "epsilon",
                           "avg_delay", "drops", "steps"])

    def log(self, episode, reward, loss, epsilon, avg_delay, drops, steps):
        self.csv.writerow([episode, f"{reward:.4f}", f"{loss:.6f}",
                           f"{epsilon:.4f}", f"{avg_delay:.3f}", drops, steps])
        self.f.flush()

    def close(self):
        self.f.close()


# ── Per-agent train loops ───────────────────────────────────────────────────

def train_dqn(cfg: ExperimentConfig, seed: int) -> list:
    np.random.seed(seed)
    env   = NetworkRoutingEnv(use_mm1=cfg.env.use_mm1, failure_prob=0.002, mean_load=0.3)
    agent = DQNAgent(
        state_dim=OBS_DIM, action_dim=env.action_space.n,
        lr=cfg.dqn.lr, gamma=cfg.dqn.gamma,
        epsilon_decay=cfg.dqn.epsilon_decay,
        batch_size=cfg.dqn.batch_size, target_update=cfg.dqn.target_update,
        buffer_capacity=cfg.dqn.buffer_cap,
    )
    logger = CSVLogger(os.path.join(LOGS_DIR, f"dqn_seed{seed}.csv"))
    rewards, total_loss = [], 0.0
    for ep in range(1, cfg.train.num_episodes + 1):
        fp, ml = apply_curriculum(env, ep, cfg.train.curriculum)
        obs, _ = env.reset()
        done, total_r, steps = False, 0.0, 0
        while not done:
            action = agent.select_action(obs, valid_actions(env))
            nobs, r, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store(obs, action, r, nobs, float(done))
            total_loss += agent.learn()
            obs = nobs; total_r += r; steps += 1
        agent.decay_epsilon()
        if ep % cfg.dqn.target_update == 0:
            agent.sync_target()
        rewards.append(total_r)
        avg_l = total_loss / max(steps, 1)
        logger.log(ep, total_r, avg_l, agent.epsilon,
                   info.get("avg_delay", 0), info.get("drops", 0), steps)
        total_loss = 0.0
        if ep % cfg.train.print_every == 0:
            print(f"    [DQN s{seed}] ep={ep:4d}  "
                  f"avg_r={np.mean(rewards[-50:]):+.1f}  ε={agent.epsilon:.3f}")
    logger.close()
    agent.save(os.path.join(MODELS_DIR, f"dqn_seed{seed}.pth"))
    return rewards


def train_rainbow(cfg: ExperimentConfig, seed: int) -> list:
    np.random.seed(seed)
    env   = NetworkRoutingEnv(use_mm1=cfg.env.use_mm1, failure_prob=0.002, mean_load=0.3)
    agent = RainbowAgent(
        state_dim=OBS_DIM, action_dim=env.action_space.n,
        lr=cfg.rainbow.lr, gamma=cfg.rainbow.gamma,
        n_step=cfg.rainbow.n_step,
        epsilon_decay=cfg.rainbow.epsilon_decay,
        batch_size=cfg.rainbow.batch_size,
        per_alpha=cfg.rainbow.per_alpha,
        per_beta_start=cfg.rainbow.per_beta_start,
        buffer_cap=cfg.rainbow.buffer_cap,
        hidden=cfg.rainbow.hidden,
    )
    logger  = CSVLogger(os.path.join(LOGS_DIR, f"rainbow_seed{seed}.csv"))
    rewards = []
    for ep in range(1, cfg.train.num_episodes + 1):
        apply_curriculum(env, ep, cfg.train.curriculum)
        obs, _ = env.reset()
        done, total_r, steps, total_loss = False, 0.0, 0, 0.0
        while not done:
            action = agent.select_action(obs, valid_actions(env))
            nobs, r, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store(obs, action, r, nobs, float(done))
            total_loss += agent.learn()
            obs = nobs; total_r += r; steps += 1
        agent.decay_epsilon()
        if ep % cfg.rainbow.target_update == 0:
            agent.sync_target()
        rewards.append(total_r)
        logger.log(ep, total_r, total_loss / max(steps, 1), agent.epsilon,
                   info.get("avg_delay", 0), info.get("drops", 0), steps)
        if ep % cfg.train.print_every == 0:
            print(f"    [Rainbow s{seed}] ep={ep:4d}  "
                  f"avg_r={np.mean(rewards[-50:]):+.1f}  ε={agent.epsilon:.3f}")
    logger.close()
    agent.save(os.path.join(MODELS_DIR, f"rainbow_seed{seed}.pth"))
    return rewards


def train_gnn(cfg: ExperimentConfig, seed: int) -> list:
    np.random.seed(seed)
    env   = NetworkRoutingEnv(use_mm1=cfg.env.use_mm1, failure_prob=0.002, mean_load=0.3)
    agent = GNNAgent(
        graph=env.G, edge_list=env.edge_list,
        lr=cfg.gnn.lr, gamma=cfg.gnn.gamma,
        epsilon_decay=cfg.gnn.epsilon_decay,
        batch_size=cfg.gnn.batch_size,
        buffer_cap=cfg.gnn.buffer_cap,
        target_update=cfg.gnn.target_update,
    )
    logger  = CSVLogger(os.path.join(LOGS_DIR, f"gnn_seed{seed}.csv"))
    rewards = []
    for ep in range(1, cfg.train.num_episodes + 1):
        apply_curriculum(env, ep, cfg.train.curriculum)
        obs, _ = env.reset()
        done, total_r, steps, total_loss = False, 0.0, 0, 0.0
        while not done:
            cur, dst = env._current_node, env._dest_node
            a_idx, nbrs = agent.select_action(env, cur, dst)
            nobs, r, term, trunc, info = env.step(a_idx)
            done = term or trunc
            s_t  = (env.G, cur, dst, nbrs)
            ncur = env._current_node
            ns_t = (env.G, ncur, dst, env._get_sorted_neighbors(ncur))
            agent.store(s_t, a_idx, r, ns_t, float(done))
            total_loss += agent.learn()
            obs = nobs; total_r += r; steps += 1
        agent.decay_epsilon()
        agent.sync_target()
        rewards.append(total_r)
        logger.log(ep, total_r, total_loss / max(steps, 1), agent.epsilon,
                   info.get("avg_delay", 0), info.get("drops", 0), steps)
        if ep % cfg.train.print_every == 0:
            print(f"    [GNN s{seed}] ep={ep:4d}  "
                  f"avg_r={np.mean(rewards[-50:]):+.1f}  ε={agent.epsilon:.3f}")
    logger.close()
    agent.save(os.path.join(MODELS_DIR, f"gnn_seed{seed}.pth"))
    return rewards


def train_qrouting(cfg: ExperimentConfig, seed: int) -> list:
    """Q-routing doesn't use a neural net — runs faster."""
    np.random.seed(seed)
    env   = NetworkRoutingEnv(use_mm1=cfg.env.use_mm1, failure_prob=0.002, mean_load=0.3)
    env.reset(seed=seed)
    agent = QRoutingAgent(env.G, lr=cfg.q_routing.lr, init_val=cfg.q_routing.init_val)
    logger  = CSVLogger(os.path.join(LOGS_DIR, f"qrouting_seed{seed}.csv"))
    rewards = []
    for ep in range(1, cfg.train.num_episodes + 1):
        apply_curriculum(env, ep, cfg.train.curriculum)
        env.reset()
        src, dst = env._current_node, env._dest_node
        env._update_link_conditions()
        path, delay, reached = agent.route_episode(env, src, dst, max_hops=30)
        reward = (20.0 if reached else -10.0) - delay / 20.0
        rewards.append(reward)
        logger.log(ep, reward, 0.0, 0.0, delay, 0, len(path))
        if ep % cfg.train.print_every == 0:
            print(f"    [QR s{seed}] ep={ep:4d}  "
                  f"avg_r={np.mean(rewards[-50:]):+.1f}")
    logger.close()
    agent.save(os.path.join(MODELS_DIR, f"qrouting_seed{seed}.json"))
    return rewards


# ── Reward curve plot ──────────────────────────────────────────────────────

def plot_all_reward_curves(all_rewards: dict, out_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {"dqn": "steelblue", "rainbow": "crimson",
              "gnn": "darkorange", "qrouting": "green"}
    W = 30

    for agent_name, runs in all_rewards.items():
        if not runs:
            continue
        min_len = min(len(r) for r in runs)
        arr    = np.array([r[:min_len] for r in runs])
        mean_r = arr.mean(0)
        std_r  = arr.std(0)
        n_ep   = len(mean_r)

        # Smooth
        smooth_mean = np.convolve(mean_r, np.ones(W)/W, mode="valid")
        smooth_std  = np.convolve(std_r,  np.ones(W)/W, mode="valid")
        x = np.arange(len(smooth_mean))

        c = colors.get(agent_name, "gray")
        ax.plot(x, smooth_mean, label=agent_name.upper(), color=c, linewidth=2)
        ax.fill_between(x, smooth_mean - smooth_std, smooth_mean + smooth_std,
                        alpha=0.15, color=c)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward ({W}-ep MA ± 1 std across seeds)")
    ax.set_title("Training Reward Curves — All Agents")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Reward curves saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default=None)
    parser.add_argument("--agents",   nargs="+",
                        default=["dqn", "rainbow", "gnn", "qrouting"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds",    type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else ExperimentConfig()
    if args.episodes:
        cfg.train.num_episodes = args.episodes
    if args.seeds:
        cfg.train.num_seeds = args.seeds

    print(f"Training {args.agents} | {cfg.train.num_episodes} ep × {cfg.train.num_seeds} seeds")
    print(f"Curriculum stages: {cfg.train.curriculum}")

    all_rewards = {}
    trainers = {
        "dqn":      train_dqn,
        "rainbow":  train_rainbow,
        "gnn":      train_gnn,
        "qrouting": train_qrouting,
    }
    t0 = time.time()

    for agent_name in args.agents:
        if agent_name not in trainers:
            print(f"Unknown agent: {agent_name}")
            continue
        print(f"\n=== Training {agent_name.upper()} ===")
        runs = []
        for seed in range(cfg.train.num_seeds):
            print(f"  Seed {seed}/{cfg.train.num_seeds-1}")
            rewards = trainers[agent_name](cfg, seed)
            runs.append(rewards)
        all_rewards[agent_name] = runs
        tail = [np.mean(r[int(len(r)*0.8):]) for r in runs]
        print(f"  → {agent_name} final: {np.mean(tail):+.2f} ± {np.std(tail):.2f}")

    # Save best DQN / Rainbow weights for evaluation
    for agent_name in args.agents:
        seeds_paths = [os.path.join(MODELS_DIR, f"{agent_name}_seed{s}.pth")
                       for s in range(cfg.train.num_seeds)]
        best_path = os.path.join(MODELS_DIR, f"{agent_name}_trained.pth")
        if seeds_paths and os.path.exists(seeds_paths[0]):
            import shutil
            shutil.copy(seeds_paths[0], best_path)

    # Keep dqn_trained.pth for backward compat (always refreshed)
    import shutil
    for alias in [("dqn", "dqn_trained.pth"), ("rainbow", "rainbow_trained.pth")]:
        src_seed0 = os.path.join(MODELS_DIR, f"{alias[0]}_seed0.pth")
        dst_alias = os.path.join(MODELS_DIR, alias[1])
        if os.path.exists(src_seed0):
            shutil.copy(src_seed0, dst_alias)

    plot_all_reward_curves(
        all_rewards,
        os.path.join(RESULTS_DIR, "training_reward.png"),
    )

    elapsed = time.time() - t0
    print(f"\nTotal training time: {elapsed:.0f}s")
    print(f"Models saved → {MODELS_DIR}/")
    print(f"Logs   saved → {LOGS_DIR}/")


if __name__ == "__main__":
    main()
