# AI-Assisted Adaptive Routing using Reinforcement Learning

A Computer Networks project demonstrating how a Deep Q-Network (DQN) agent learns
to route packets optimally under dynamic traffic loads and random link failures —
outperforming static algorithms (Dijkstra, Random) especially under failure conditions.

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
│   └── network_env.py          Custom Gymnasium environment
├── agents/
│   └── dqn_agent.py            DQN with experience replay + target net
├── baselines/
│   ├── dijkstra.py             Dijkstra shortest-path (OSPF analog)
│   └── random_routing.py       Random / flooding baseline
├── real_network/
│   ├── probe_server.py         UDP echo server (runs on each real node)
│   ├── probe_client.py         Sends probes, measures RTT/loss
│   ├── real_env_adapter.py     Maps real probe metrics to the agent state vector
│   ├── routing_controller.py   Control plane: loads model, makes routing decisions
│   ├── dashboard.py            Rich CLI live decision dashboard
│   ├── forwarder.py            Data plane: per-node source-routing packet forwarder
│   ├── data_plane.py           Controller that forwards real bytes + reroutes on failure
│   └── virtual_demo.py         Single-laptop end-to-end data delivery + failure demo
├── models/
│   └── dqn_trained.pth         Saved after training
├── results/                    Auto-created — plots land here
├── train.py                    Training script (simulated)
├── evaluate.py                 Comparison & plots
├── visualize.py                Topology & heatmap
├── simulation_dashboard.py     Web dashboard (validated results + live animation)
├── requirements.txt
├── README.md
├── PRESENTATION_GUIDE.md       Demo script + readiness assessment
└── setup_real_network.md       Step-by-step real network guide
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

## CN Concepts Demonstrated

| Concept | Where |
|---------|-------|
| OSPF link-state metric | `build_weighted_graph()` in dijkstra.py |
| TCP congestion window backoff | `DROP_PENALTY` in network_env.py |
| TTL / loop prevention | `LOOP_PENALTY` + visited-set in network_env.py |
| ICMP echo / ping | UDP probe packets in probe_server/client |
| BGP route flapping | Random link failure/recovery in `_update_link_conditions()` |
| SDN control plane | routing_controller.py + dashboard.py |
| RED / ECN congestion signal | loss_rate = f(congestion) in network_env.py |

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.0
- Gymnasium ≥ 0.29
- NetworkX, Matplotlib, Seaborn, NumPy, Rich
