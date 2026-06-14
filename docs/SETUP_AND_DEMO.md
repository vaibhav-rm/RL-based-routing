# Setup & Demonstration Guide

How to install the project and demonstrate it two ways:

1. **Simulation** — everything on one laptop, zero hardware, ~2 minutes to a live demo.
2. **Hardware** — the agent driving real packet forwarding across 3–4 devices on a LAN.

> For *how the project works internally*, see [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md).

---

## 0. Install (once)

```bash
git clone <repo> && cd RL_based_routing
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest          # sanity check: 73 tests pass in ~5 s
```

Pre-trained models (`models/`) and results (`results/`) are committed, so every
demo below runs **without retraining**. To retrain from scratch:
`.venv/bin/python train.py` (~3–8 min on CPU).

> **venv + ROS gotcha:** if `pytest` aborts during collection, the venv is
> inheriting system ROS 2 plugins. `pytest.ini` already disables them; if needed,
> run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest`.

---

## 1. Simulation demo (one laptop, no hardware)

### 1a. Web dashboard — the headline visual

```bash
.venv/bin/python simulation_dashboard.py
# open http://127.0.0.1:5000  (full-screen on the projector)
```

To drive it from another device on the same Wi-Fi (e.g. your phone):

```bash
.venv/bin/python simulation_dashboard.py --host 0.0.0.0 --port 5000
# open http://<laptop-ip>:5000
```

On screen:
* **Left — Packet Delivery Ratio + delay bars.** The *real, validated* numbers
  from `results/evaluation_results.json`. The 0/20/40/60 % buttons switch the
  link-failure rate.
* **Right — live topology animation.** A GNN-DQN packet hops across the network
  while links fail, recover and change congestion. Clearly labelled *illustrative*
  (it builds intuition for the validated bars on the left).

### 1b. Real data delivery + reroute on failure — the most convincing command

```bash
.venv/bin/python -m network_rl.real_network.virtual_demo --src 0 --dst 9
```

This spawns one forwarder per node on localhost, delivers a **real message**
end-to-end, then **kills a relay node and shows the controller reroute and
re-deliver** the payload — with an integrity check. (Defaults to the GNN; add
`--agent dqn` to use the DQN.)

### 1c. Regenerate any result or figure

```bash
.venv/bin/python evaluate.py                       # 7-algorithm comparison
.venv/bin/python experiments/scalability.py        # zero-shot to 50 nodes
.venv/bin/python experiments/adversarial.py        # targeted vs random failures
.venv/bin/python experiments/pareto.py             # latency/reliability fronts
.venv/bin/python experiments/continual_learning.py # catastrophic forgetting
.venv/bin/python visualize.py                      # topology + congestion heatmaps
```

Outputs land in `results/` (JSON + PNG). These feed the paper and deck directly.

### 1d. Suggested 3-minute script

1. Open the **web dashboard**; click 0 % → 60 %. Talking point: *GNN-DQN is the
   only method at 100 % delivery across every failure rate, and it also has the
   lowest delay (~10 ms).*
2. Be upfront: *vanilla DQN/Rainbow are weaker (0.83–0.93) — that motivates the
   GNN and Rainbow additions.* A demo that admits weak baselines is credible.
3. Run **`virtual_demo`**: show real bytes crossing the network, kill a relay,
   watch the controller reroute and still deliver.
4. (Optional) Open `results/continual_learning.png`: *the new result — naive
   training forgets the old regime; rehearsal fixes it.*

> Honesty note baked into the dashboard: on a *static* snapshot Dijkstra is
> delay-optimal by definition, so we deliberately do **not** stage a misleading
> "fail one link, watch the baseline collapse" race. The trustworthy comparison
> is the validated batch; the animation is labelled illustrative.

---

## 2. Hardware demo (3–4 devices on a LAN)

Shows the **control plane** (agent decides next hop from live probe metrics) and
the **data plane** (real bytes forwarded, surviving a node failure) on real
machines.

### 2a. What you need

| Role | Count | Examples |
|------|-------|----------|
| Routing controller | 1 | Laptop running the dashboard + agent |
| Network nodes | 2–3 | Laptops / Raspberry Pi 4 / VMs |

All devices on the **same Layer-2 segment** (same Wi-Fi AP or switch).

### 2b. Assign static IPs (192.168.10.0/24)

```bash
# Linux / Raspberry Pi (per device)
sudo ip addr add 192.168.10.1/24 dev eth0   # Node 0 (controller)
sudo ip addr add 192.168.10.2/24 dev eth0   # Node 1
sudo ip addr add 192.168.10.3/24 dev eth0   # Node 2
sudo ip addr add 192.168.10.4/24 dev eth0   # Node 3
```

```powershell
# Windows (admin PowerShell)
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.10.2 -PrefixLength 24 -DefaultGateway 192.168.10.1
```

```bash
# macOS
sudo ifconfig en0 192.168.10.3 netmask 255.255.255.0
```

Verify connectivity: `ping 192.168.10.1` from every device before continuing.

### 2c. Install on every node

```bash
.venv/bin/pip install -r requirements.txt     # controller (full)
pip install numpy rich                          # probe-only nodes (minimal)
```

### 2d. Control plane — live probing + agent decisions

On **each network node** (use `tmux`/`screen` to keep them running):

```bash
python network_rl/real_network/probe_server.py --host 0.0.0.0 --port 9999 --node-id 1
# ...node-id 2 on 192.168.10.3, node-id 3 on 192.168.10.4
```

Verify probing from the **controller** (Node 0):

```bash
python network_rl/real_network/probe_client.py \
    --targets 192.168.10.2:9999 192.168.10.3:9999 192.168.10.4:9999 --count 20
```

Launch the **live CLI dashboard** on the controller:

```bash
python network_rl/real_network/dashboard.py \
    --model models/dqn_trained.pth \
    --node-map 0:192.168.10.1 1:192.168.10.2 2:192.168.10.3 3:192.168.10.4 \
    --source 0 --dest 2 --interval 3
```

It shows live link latency/loss/congestion, the **agent vs Dijkstra next-hop**,
and a rolling event log. The interesting moment is **DIVERGE**: a link is
numerically shortest but the agent has learned to avoid it from congestion
patterns — adaptive routing beyond Dijkstra.

Generate realistic load to trigger a reroute (raises congestion on 0→2):

```bash
# on Node 2
iperf3 -s -p 5201
# on the controller
iperf3 -c 192.168.10.3 -p 5201 -t 30 -b 10M
```

### 2e. Data plane — forward real bytes, survive a failure

On **every node**, also run a forwarder (alongside the probe server):

```bash
python network_rl/real_network/forwarder.py --label 1 --host 0.0.0.0 --port 9000
```

On the **controller**, `DataPlaneController` computes an agent-guided path,
source-routes the payload through the forwarders, and reroutes on any `link_down`
NACK or timeout (see the snippet in `PRESENTATION_GUIDE.md`, Part 2). Then
**power off / unplug a relay node and resend** — the data still arrives via a new
route, because the controller marks the failed edge, recomputes the agent's best
path over surviving links, and resends.

### 2f. No spare machines? Use network namespaces

```bash
sudo ip netns add node0 && sudo ip netns add node1
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 netns node0 && sudo ip link set veth1 netns node1
sudo ip netns exec node0 ip addr add 192.168.10.1/24 dev veth0 && sudo ip netns exec node0 ip link set veth0 up
sudo ip netns exec node1 ip addr add 192.168.10.2/24 dev veth1 && sudo ip netns exec node1 ip link set veth1 up
sudo ip netns exec node1 python network_rl/real_network/probe_server.py --port 9999 &
sudo ip netns exec node0 python network_rl/real_network/probe_client.py --targets 192.168.10.2:9999 --count 5
```

---

## 3. Reading the live output

| Field | Meaning |
|-------|---------|
| RTT (ms) | Round-trip time from the UDP probe |
| Loss % | % of probes that timed out (link quality) |
| Congestion bar | Normalized [0–1] metric fed to the agent |
| Agent next-hop | Node the policy recommends forwarding to |
| Dijkstra next-hop | Shortest-path recommendation (for comparison) |
| AGREE / DIVERGE | Whether the two chose the same next hop |

---

## 4. Honest "how far from production" framing (for Q&A)

* **What is real:** the agents are genuinely trained; evaluation numbers are
  validated batches; the data plane forwards real bytes and reroutes on failure.
* **What is simulated:** failure injection, congestion dynamics, and the M/M/1
  delay model. Failed links are penalized but traversable in the base env, so PDR
  saturates — the discriminating metrics (delay stretch, targeted-failure
  delivery, fairness, forgetting) carry the analysis.
* **What is not done yet:** a multi-node hardware testbed at scale, hard-cut
  failure semantics in production, and decentralized multi-agent control. These
  are stated as future work in the paper, not glossed over.

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `pytest` aborts during collection | ROS plugins; `pytest.ini` disables them, or set `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. |
| Probe client times out | Check firewall on the probe node; confirm `ping` works; ensure `--port` matches. |
| Dashboard shows all AGREE | Add load with `iperf3` so a link congests enough for the agent to DIVERGE. |
| Model not found | Run `.venv/bin/python train.py` to (re)create `models/`. |
| `virtual_demo` port in use | A previous run left forwarders up; kill stale `python` processes or change `--src/--dst`. |
