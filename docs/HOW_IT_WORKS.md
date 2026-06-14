# How This Project Works

A technical walkthrough of the RL-based adaptive routing project: what each part
does, how data flows from the simulator to a trained policy to the results and
the paper, and how to reproduce everything.

> For installing and *demonstrating* the project (simulation and hardware), see
> [`SETUP_AND_DEMO.md`](SETUP_AND_DEMO.md). This document explains the internals.

---

## 1. The big picture

The project answers one question: **can a reinforcement-learning agent learn to
forward packets better than classical shortest-path routing under realistic,
changing network conditions?** — and then stress-tests that policy along axes a
single accuracy number hides (generalization, trade-offs, robustness, fairness,
continual adaptation).

```
            ┌────────────────────────┐
            │  NetworkRoutingEnv      │  10-node graph, M/M/1 delay,
            │  (Gymnasium env)        │  AR(1) congestion, Markov failures
            └───────────┬────────────┘
                        │ observation (70-dim) / reward
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Learners        Baselines        Experiments
   DQN / Rainbow   Dijkstra         scalability, pareto,
   GNN-DQN /       ECMP             adversarial, fairness,
   Q-Routing       Random           continual_learning
        │               │               │
        └───────────────┴───────────────┘
                        ▼
                 results/*.json + *.png
                        ▼
              paper/ (docx) + pitch deck
```

There are **two planes**, mirroring real SDN:

* **Control plane** — the agent *decides* the next hop. This is what training and
  all the simulated experiments exercise.
* **Data plane** — real bytes are *forwarded* across nodes and survive failures
  (`network_rl/real_network/`). Used for the live/hardware demo.

---

## 2. The environment (`network_rl/env/network_env.py`)

`NetworkRoutingEnv` is a custom [Gymnasium](https://gymnasium.farama.org/) env.

* **Topology** — 10 nodes, 17 bidirectional links; each link has a bandwidth and
  a base propagation delay (`TOPOLOGY_EDGES`).
* **Delay model** — M/M/1 queuing (`mm1_delay`): `W = base_delay + tx/(1 − ρ)`
  where `ρ` is utilization. Delay explodes as a link saturates, exactly like a
  real queue (Kleinrock, 1975).
* **Dynamics** (`_update_link_conditions`, run every step):
  * congestion follows an **AR(1)** random walk with bursty injections;
  * loss rate rises with utilization in a **RED-like** curve;
  * links **fail and recover** under a two-state Markov chain (`failure_prob`).
* **Observation** (`_get_obs`) — a 70-dim vector: 4 features per link
  `[congestion, normalized delay, loss, up/down]` (17 × 4 = 68) plus the
  normalized current and destination node IDs.
* **Action** — pick one of the current node's sorted neighbours (next hop).
* **Reward** (`step`) — delivery bonus on reaching the destination; per-hop
  delay penalty scaled by `w_delay`; a drop penalty `w_drop` when a lossy link
  drops; loop and invalid-action penalties. `w_delay`/`w_drop` are exposed so the
  latency/reliability trade-off can be swept (see the Pareto experiment).
* **Curriculum** — `set_curriculum(failure_prob, mean_load)` ramps difficulty
  during training.

The env is the single source of truth: agents and baselines all see the same
graph state, so comparisons are apples-to-apples.

---

## 3. The agents (`network_rl/agents/`)

| Agent | File | Idea |
|-------|------|------|
| **DQN** | `dqn_agent.py` | Double-DQN, ε-greedy, replay buffer, target net. The MLP reads the 70-dim observation. |
| **Rainbow** | `rainbow_agent.py` | DQN + Dueling heads + Prioritized Experience Replay + n-step returns. |
| **GNN-DQN** | `gnn_agent.py` | GraphSAGE-style message passing over the *live link features*. Q-value per (current, neighbour) pair. **Topology-agnostic** — weights depend on feature dimensions, not node count, so it runs on any graph. |
| **Q-Routing** | `q_routing.py` | Classic distributed tabular RL (Boyan & Littman, 1994); per-node Q-tables updated online. |
| **Continual DQN** | `continual_dqn_agent.py` | DQN + optional Elastic Weight Consolidation, via a minimal `_extra_loss` hook in `DQNAgent`. Used only by the continual-learning study. |

Classical **baselines** (`network_rl/baselines/`): `dijkstra.py` (OSPF analog,
shortest path on the live delay metric), `ecmp.py` (equal-cost multi-path load
balancing), `random_routing.py` (uniform next hop).

**Inference-determinism note.** All neural agents share one subtle rule: the
"act randomly until the replay buffer fills" warm-up applies *only during
training* (`policy_net.training`). A loaded, eval-mode agent at ε = 0 reads its
Q-values deterministically. The test suite guards this
(`test_*_eval_mode_is_deterministic_not_random`) because an earlier bug made
every evaluation secretly route randomly.

---

## 4. Training (`train.py`)

```bash
.venv/bin/python train.py --agents dqn rainbow gnn qrouting --episodes 800 --seeds 5
```

* Trains each agent for N episodes × M seeds, with curriculum-increasing
  difficulty.
* Writes per-episode CSV logs to `logs/` and model weights to `models/`
  (`*_seed{0..4}.pth`, Q-tables as `*.json`).
* Plots smoothed multi-seed reward curves to `results/training_reward.png`.

Config lives in `experiments/config.py` (or a YAML passed with `--config`).

---

## 5. Evaluation (`evaluate.py`)

```bash
.venv/bin/python evaluate.py        # 7 algorithms × 4 failure rates
```

* Loads the trained models, runs greedy policies (ε = 0) over many episodes at
  failure rates 0/20/40/60 %.
* Records **PDR** (packet delivery ratio), **end-to-end delay**, **jitter** and a
  **throughput** proxy.
* Saves `results/evaluation_results.json` and the publication plot
  `results/evaluation_comparison.png`.
* Statistical helpers in `network_rl/analysis/statistics.py` provide bootstrap
  CIs, Welch's t-test, Cohen's d, IQM, Jain's fairness index and Pareto-front
  detection.

> **Failure semantics caveat (important for reading results).** A failed link is
> *penalized but still traversable* in the base env. So PDR measures successful
> navigation within a hop budget and saturates near 1.0 for good policies — which
> is why the research experiments below add more discriminating metrics (delay
> stretch, targeted-failure delivery, fairness, forgetting).

---

## 6. The research experiments (`experiments/`)

Each script loads the trained models (or trains fresh where noted), produces a
JSON + a plot in `results/`, and is covered by tests.

| Script | Question | Headline finding |
|--------|----------|------------------|
| `generalisation_test.py` | Transfer to a slightly perturbed graph (+4 edges)? | GNN transfers; fixed-input MLPs can't even run on the new obs size. |
| `scalability.py` | Zero-shot transfer to *much larger* graphs (20–50 nodes)? | GNN holds delay-stretch ≈ 1.1–1.2; random walk blows up to 10.7. |
| `pareto.py` | Where is the latency/reliability trade-off? | GNN-DQN is the **sole Pareto-optimal** algorithm; plus a within-learner front from sweeping `w_drop`. |
| `adversarial.py` | Worst-case vs random failures? | GNN keeps PDR 1.0; vanilla DQN collapses 0.91 → 0.60 under targeted attack. |
| `fairness_eval.py` | Does the mean hide per-flow starvation? | DQN & Q-Routing starve some flows (Jain 0.92, min-PDR 0); GNN/Dijkstra/ECMP fair. |
| `continual_learning.py` | Forgetting when adapting to a new regime? | Naive forgets (0.77 → 0.09); **rehearsal** fixes it; **EWC** does not transfer here. |
| `sweep.py`, `ablation.py` | Hyperparameter sweep / Rainbow component ablation | Reproducibility + component contributions. |

Run any of them, e.g.:

```bash
.venv/bin/python experiments/scalability.py
.venv/bin/python experiments/continual_learning.py
```

---

## 7. Data plane — moving real bytes (`network_rl/real_network/`)

The simulator is the control plane. The data plane proves the agent can drive
*real* forwarding that survives failures:

* `forwarder.py` — a dumb, topology-agnostic source-routing daemon (one per
  node). The full route `[label, ip, port]` is embedded in the packet header;
  a node that can't reach its next hop NACKs the controller.
* `data_plane.py` — `DataPlaneController` computes an agent-guided route (GNN by
  default), source-routes the payload, and **reroutes around failures** with a
  retry budget.
* `virtual_demo.py` — single-laptop end-to-end demo: spawns a forwarder per node
  on localhost, delivers a message, kills a relay node, and shows the controller
  reroute and re-deliver with a payload integrity check.
* `probe_server.py` / `probe_client.py` — UDP latency/loss probing for live link
  metrics. `routing_controller.py` / `dashboard.py` drive the control-plane
  decisions on real hardware.

This is **TRL ≈ 4**: both planes demonstrated, on localhost and on a small LAN.

---

## 8. From results to the paper

* `simulation_dashboard.py` — a self-contained Flask dashboard: validated PDR/delay
  bars (from `evaluation_results.json`) plus a live illustrative GNN animation.
* `paper/build_paper.py` — regenerates the IEEE conference paper
  (`paper/RL_Adaptive_Routing.docx`) from the template, embedding the real figures
  and numbers from `results/`.
* `paper/build_deck.py` — regenerates the pitch deck (`paper/RL_Routing_Pitch.pptx`).

Both read straight from `results/`, so re-running the experiments and then the
build scripts keeps the paper and deck in sync with the data.

---

## 9. Repository map

```
network_rl/
  env/           NetworkRoutingEnv, M/M/1 delay, traffic model
  agents/        dqn, rainbow, gnn, q_routing, per_buffer, continual_dqn
  baselines/     dijkstra, ecmp, random_routing
  analysis/      statistics, convergence, explain
  real_network/  forwarder, data_plane, probe_*, routing_controller, dashboard, virtual_demo
experiments/     config, sweep, ablation, generalisation_test, fairness_eval,
                 scalability, adversarial, pareto, continual_learning
paper/           build_paper.py, build_deck.py, ieee_template.docx, outputs
docs/            HOW_IT_WORKS.md (this file), SETUP_AND_DEMO.md
tests/           73 pytest cases (env, agents, baselines, stats, experiments)
train.py         multi-seed / curriculum training
evaluate.py      7-algorithm comparison + plots
visualize.py     topology & congestion heatmaps
simulation_dashboard.py   web dashboard
models/  logs/  results/   trained weights, CSV logs, JSON + plots
```

---

## 10. Reproduce everything

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                       # 73 tests, ~5 s
.venv/bin/python train.py                         # (re)train agents → models/
.venv/bin/python evaluate.py                      # core comparison → results/
.venv/bin/python experiments/scalability.py       # each study writes results/
.venv/bin/python experiments/adversarial.py
.venv/bin/python experiments/pareto.py
.venv/bin/python experiments/fairness_eval.py
.venv/bin/python experiments/continual_learning.py
.venv/bin/python paper/build_paper.py             # → paper/RL_Adaptive_Routing.docx
.venv/bin/python paper/build_deck.py              # → paper/RL_Routing_Pitch.pptx
```

> Pre-trained models and committed results are already in the repo, so the
> experiments and the paper build run without retraining.
