# AI-Assisted Adaptive Routing using Reinforcement Learning

A Computer Networks project demonstrating how a Deep Q-Network (DQN) agent learns
to route packets optimally under dynamic traffic loads and random link failures —
outperforming static algorithms (Dijkstra, Random) especially under failure conditions.

**Documentation:**
- [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) — technical walkthrough of every component and the data flow.
- [`docs/SETUP_AND_DEMO.md`](docs/SETUP_AND_DEMO.md) — install + demo (simulation on one laptop, and on hardware).
- [`paper/`](paper/) — the IEEE conference paper (`RL_Adaptive_Routing.docx`) and pitch deck (`RL_Routing_Pitch.pptx`), regenerable via `build_paper.py` / `build_deck.py`.

---

## ASCII Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SIMULATED MODE                                │
│                                                                      │
│  NetworkRoutingEnv (Gymnasium)                                       │
│  ┌─────────────────────────────────────────────┐                    │
│  │  10-node graph   M/M/1 queuing delay        │                    │
│  │  17 edges        AR(1) congestion + failures │                    │
│  │  State: 70-dim = [cong, delay, loss, up]×17  │                    │
│  │         + [cur_node, dst_node]   (4/edge + 2)│                    │
│  │  Action: next-hop index (0..max_degree-1)   │                    │
│  │  Reward: -delay - drop_penalty - loop_pen   │                    │
│  └───────────────┬─────────────────────────────┘                    │
│                  │ obs / reward                                       │
│                  ▼                                                   │
│  ┌──────────────────────────────────────────┐                       │
│  │  DRL Agents (PyTorch)                    │                       │
│  │  Rainbow (Dueling+PER+n-step+Double)     │                       │
│  │  GNN-DQN (topology-agnostic)             │                       │
│  │  Vanilla DQN  •  Q-Routing (tabular)     │                       │
│  │  ReplayBuffer (50k transitions)          │  → models/*.pth       │
│  └──────────────────────────────────────────┘                       │
│                                                                      │
│  Baselines:  Dijkstra (OSPF-like) | ECMP | Random (flooding)        │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        REAL NETWORK MODE                             │
│                                                                      │
│  Physical/VM Nodes                                                   │
│  ┌──────────┐    UDP probes    ┌──────────┐  ┌──────────┐           │
│  │ Node 0   │ ─────────────── │ Node 1   │  │ Node 2   │           │
│  │ (ctrl)   │ ◄──────────────  │ probe_  │  │ probe_  │           │
│  │          │  RTT + loss%     │ server  │  │ server  │           │
│  └────┬─────┘                  └──────────┘  └──────────┘           │
│       │                                                              │
│  ┌────▼──────────────────────────────────────────────────┐         │
│  │  RealEnvAdapter                                        │         │
│  │  RTT/loss → normalised congestion vector               │         │
│  │  Builds same state shape as simulated env              │         │
│  └────┬───────────────────────────────────────────────────┘        │
│       │                                                              │
│  ┌────▼──────────────────────────────────────────────────┐         │
│  │  DQN Agent (loaded from models/dqn_trained.pth)        │         │
│  │  → recommends next-hop                                 │         │
│  │  Dijkstra on same graph for comparison                 │         │
│  └────┬───────────────────────────────────────────────────┘        │
│       │                                                              │
│  ┌────▼───────────────────┐                                         │
│  │  Rich CLI Dashboard     │  latency | loss | DQN vs Dijkstra      │
│  └────────────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
network_rl/
├── env/
│   ├── network_env.py          Custom Gymnasium environment (M/M/1, AR(1), failures)
│   └── traffic_model.py        Self-similar (Pareto ON/OFF) background traffic
├── agents/
│   ├── dqn_agent.py            Vanilla DQN (Double-DQN + target net + replay)
│   ├── rainbow_agent.py        Rainbow: Dueling + PER + n-step + Double
│   ├── gnn_agent.py            GNN-DQN (topology-agnostic message passing)
│   ├── q_routing.py            Q-Routing (Boyan & Littman 1994, distributed)
│   └── per_buffer.py           Prioritised experience replay buffer
├── baselines/
│   ├── dijkstra.py             Dijkstra shortest-path (OSPF analog)
│   ├── ecmp.py                 Equal-Cost Multi-Path (RFC 2991/2992)
│   └── random_routing.py       Random / flooding baseline
├── analysis/
│   ├── statistics.py           Bootstrap CIs, Welch t-test, Cohen's d, IQM
│   └── convergence.py          Convergence episode / sample-efficiency metrics
├── real_network/
│   ├── probe_server.py         UDP echo server (runs on each real node)
│   ├── probe_client.py         Sends probes, measures RTT/loss
│   ├── real_env_adapter.py     Maps real probe metrics to the 70-dim state vector
│   ├── routing_controller.py   Control plane: loads model, makes routing decisions
│   ├── dashboard.py            Rich CLI live decision dashboard
│   ├── forwarder.py            Data plane: per-node source-routing packet forwarder
│   ├── data_plane.py           Controller that forwards real bytes + reroutes on failure
│   └── virtual_demo.py         Single-laptop end-to-end data delivery + failure demo
experiments/
├── config.py                   YAML experiment configuration
├── sweep.py                    Hyperparameter sweep
├── ablation.py                 Ablation study
├── generalisation_test.py      Unseen-topology transfer test (+4 edges)
├── fairness_eval.py            Per-flow Jain's fairness index
├── scalability.py              Zero-shot GNN transfer to 20–50-node graphs
├── adversarial.py              Worst-case (critical-link) vs. random failures
├── pareto.py                   Latency–reliability multi-objective Pareto fronts
└── continual_learning.py       Catastrophic forgetting + rehearsal/EWC remedies
tests/                          pytest suite (73 tests) — env, agents, baselines, stats
models/                         Trained weights (*.pth) and Q-tables (*.json)
logs/                           Per-seed training CSVs
results/                        Evaluation JSON + publication plots
archive/                        Superseded artifacts (see archive/README.md)
train.py                        Multi-seed / curriculum training
evaluate.py                     7-algorithm comparison & plots
visualize.py                    Topology & congestion heatmaps
simulation_dashboard.py         Web dashboard (validated results + live animation)
pytest.ini                      Test config
requirements.txt
README.md
PRESENTATION_GUIDE.md           Demo script + readiness assessment
setup_real_network.md           Step-by-step real network guide
```

---

## Quick Start — Simulated Mode

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the DQN (~3–8 min on CPU)
python train.py

# 3. Evaluate and plot comparisons
python evaluate.py

# 4. Visualise topology and congestion heatmap
python visualize.py

# 5. Live web dashboard (validated results + animated GNN routing)
python simulation_dashboard.py     # → http://127.0.0.1:5000
```

All output images land in `results/`. The dashboard and the full demo script for a
talk are described in [`PRESENTATION_GUIDE.md`](PRESENTATION_GUIDE.md).

---

## Quick Start — Real Network Mode

See [setup_real_network.md](setup_real_network.md) for the full walkthrough.

Short version:
```bash
# On each probe node
python network_rl/real_network/probe_server.py --port 9999

# On the controller (after training)
python network_rl/real_network/dashboard.py \
    --model models/dqn_trained.pth \
    --node-map 0:192.168.10.1 1:192.168.10.2 2:192.168.10.3 \
    --source 0 --dest 2 --interval 3
```

**Data plane (forward real bytes + reroute around a failure)** — try it on one laptop,
no hardware needed:
```bash
python -m network_rl.real_network.virtual_demo --src 0 --dst 9
```
It launches a forwarder per node, delivers a real message, then kills a relay node and
shows the controller reroute and re-deliver the data. See `setup_real_network.md`.

---

## Sample Results (Simulated Evaluation)

Packet Delivery Ratio (PDR) across link-failure rates, 300 episodes per setting.
End-to-end delay shown at 0% / 60% failure. Generated by `evaluate.py`.

| Algorithm | PDR @0% | PDR @40% | PDR @60% | Delay @0% | Delay @60% |
|-----------|:-------:|:--------:|:--------:|:---------:|:----------:|
| **GNN-DQN** | **1.00** | **1.00** | **1.00** | **10.0 ms** | **10.2 ms** |
| Rainbow   | 0.93 | 0.91 | 0.90 | 15.7 ms | 15.4 ms |
| DQN       | 0.86 | 0.85 | 0.83 | 12.3 ms | 12.8 ms |
| Dijkstra  | 0.95 | 0.99 | 0.91 | 13.9 ms | 15.1 ms |
| ECMP      | 0.95 | 1.00 | 0.92 | 12.9 ms | 12.6 ms |
| Q-Routing | 0.54 | 0.46 | 0.40 | 35.3 ms | 37.1 ms |
| Random    | 1.00 | 0.79 | 0.86 | 37.2 ms | 36.5 ms |

Key observation: **the GNN-DQN agent is the only method that maintains 100% PDR
at every failure rate while also achieving the lowest end-to-end delay** — because
it routes from graph structure (message passing over current link features) rather
than a memorised topology, it reroutes around failed links without degradation.
The strong shortest-path baselines (Dijkstra, ECMP) remain competitive at low
failure but drop to ~0.91 at 60% failure; vanilla DQN trails the baselines on PDR,
which is exactly why the dueling/PER/n-step additions (Rainbow) and the
structure-aware GNN are needed.

---

## Research Extensions

Beyond the headline comparison, four experiments probe the gaps that a single
aggregate PDR number hides. Every figure/JSON is produced from the trained
models or a real training sweep — none of the numbers are fabricated.

**Scalability — zero-shot transfer to larger graphs** (`experiments/scalability.py`).
The GNN trained on the 10-node topology is applied unchanged to Barabási–Albert
scale-free graphs of 20–50 nodes. Because PDR is uninformative on
well-connected graphs (anything delivers), the discriminating metric is *delay
stretch* (policy delay ÷ Dijkstra-optimal delay). The zero-shot GNN holds stretch
≈ **1.1–1.2** up to 50 nodes (5× training size) while a random walk degrades from
3.2 → 10.7 — the learned policy transfers route *quality*, not just reachability.

**Adversarial robustness — worst-case vs. average-case failures**
(`experiments/adversarial.py`). Instead of random link failures, a targeted
attack removes the highest edge-betweenness (most critical) links at the same
budget. The GNN keeps PDR = 1.0 under both; **vanilla DQN collapses from 0.91 to
0.60** at a 20% targeted budget — its biggest robustness weakness, invisible to
random-failure testing. (Failed links are high-penalty but traversable, the
project-wide convention; Dijkstra routes only the active subgraph and is shown
as a hard-cut reachability reference.)

**Multi-objective Pareto fronts** (`experiments/pareto.py`). Latency and
reliability are competing objectives, so a single scalarised reward hides a
trade-off. (A) Across the seven algorithms in (delay, PDR) space, **GNN-DQN is
the sole Pareto-optimal point** — lowest delay *and* highest PDR, dominating
every baseline. (B) Sweeping the reward's drop-penalty weight `w_drop` and
training a DQN per setting traces the learned policy's own latency–reliability
front (multi-seed averaged), with the non-dominated operating points identified
by `statistics.pareto_front`.

**Per-flow fairness** (`experiments/fairness_eval.py`). Jain's index over
per-(src,dst) delivery ratios exposes flow starvation that the aggregate mean
masks (e.g. DQN and Q-Routing starve some flows while keeping mean PDR high).

**Catastrophic forgetting and continual learning** (`experiments/continual_learning.py`).
A deployed agent must keep adapting, but training on a new regime overwrites the
old policy. We train one DQN sequentially on two conflicting destination tasks
(route to {7,8,9} then {0,1,2}) and measure forgetting of the first. Result
(4 seeds): naive sequential training **forgets catastrophically** — Task-A PDR
collapses 0.77 → 0.09. **Experience rehearsal** (retaining old transitions in the
replay buffer) fully prevents it (forgetting ≈ 0). EWC, the standard supervised
continual-learning remedy, **does not transfer**: because the policy is
conditioned on the destination (an input feature), both tasks share the same
weights and a diagonal-Fisher anchor cannot isolate task-specific computation —
the honest takeaway is that data-space rehearsal, nearly free in RL via the
replay buffer, is the robust fix.

```bash
python experiments/scalability.py
python experiments/adversarial.py
python experiments/pareto.py          # trains a DQN per operating point (~4 min)
python experiments/fairness_eval.py
python experiments/continual_learning.py   # sequential training, 3 methods (~4 min)
```

---

## CN Concepts Demonstrated

| Concept | Where |
|---------|-------|
| OSPF link-state metric | `build_weighted_graph()` in dijkstra.py |
| TCP congestion window backoff | `w_drop` drop penalty in network_env.py |
| TTL / loop prevention | `LOOP_PENALTY` + visited-set in network_env.py |
| ICMP echo / ping | UDP probe packets in probe_server/client |
| BGP route flapping | Random link failure/recovery in `_update_link_conditions()` |
| SDN control plane | routing_controller.py + dashboard.py |
| RED / ECN congestion signal | loss_rate = f(congestion) in network_env.py |

---

## Testing

A `pytest` suite (73 tests) covers the environment contract, the classical
baselines, the learning agents, the statistical utilities (including Pareto
dominance), the multi-objective reward weights, the scalability/adversarial
harnesses, and the continual-learning (EWC) machinery:

```bash
pip install -r requirements.txt   # includes pytest
pytest                            # ~5 s, 73 tests
```

The most important guards are
`tests/test_agents.py::test_*_eval_mode_is_deterministic_not_random`, which pin
the inference-determinism fix — a loaded agent at ε=0 must read its Q-values
deterministically rather than falling back to random routing.

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- Gymnasium ≥ 0.29
- NetworkX, Matplotlib, Seaborn, NumPy, Rich
- pytest (for the test suite)
