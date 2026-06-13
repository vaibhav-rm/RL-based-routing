# Presentation Guide — RL-Based Adaptive Routing

A practical script for demoing this project: first the **live simulation dashboard**
(runs on one laptop, zero setup), then the **hardware demo** (multiple devices), and
finally an honest **"how far from real usage"** assessment for the Q&A.

---

## Part 1 — Live Simulation Dashboard (web)

Runs entirely on your laptop. No hardware, no internet needed.

```bash
.venv/bin/python simulation_dashboard.py
# then open http://127.0.0.1:5000 in a browser (full-screen it on the projector)
```

To present from another machine on the same Wi-Fi (e.g. control from your phone):

```bash
.venv/bin/python simulation_dashboard.py --host 0.0.0.0 --port 5000
# open http://<laptop-ip>:5000
```

### What's on screen

* **Left — Packet Delivery Ratio bars.** These are the *real, validated* numbers from
  `results/evaluation_results.json` (300 episodes per setting, produced by `evaluate.py`).
  The four buttons (0 / 20 / 40 / 60%) switch the link-failure rate. The right-hand
  number on each bar is end-to-end delay in ms.
* **Right — Live topology animation.** A GNN-DQN packet hops across the 10-node network
  while links fail (greyed/dashed), recover, and change congestion (green → red). This is
  *illustrative* — it builds intuition for the numbers on the left.

### Talking points (the honest story)

1. **GNN-DQN is the only method at 100% delivery across every failure rate, *and* it has
   the lowest end-to-end delay (~10 ms).** Click through 0% → 60% to show it holds.
2. **Why it wins:** it routes from *graph structure* (message passing over live link
   features), so it anticipates congestion and reroutes around failures over a multi-step
   path — rather than recomputing a single shortest path on a stale snapshot.
3. **Vanilla DQN/Rainbow are weaker** (PDR 0.83–0.93). Be upfront about this — it motivates
   *why* the dueling/PER/n-step additions and the structure-aware GNN were needed. A demo
   that admits its weak baselines is more credible than one that claims everything wins.
4. **Dijkstra/ECMP are strong baselines** (high delivery) but pay ~40–80% more delay than
   the GNN, and Dijkstra dips to ~0.91 at the highest failure rate.

> Honesty note baked into the dashboard: on a *static* snapshot Dijkstra is delay-optimal
> by definition, so we deliberately do **not** stage a misleading "fail one link, watch the
> baseline collapse" live race. The trustworthy comparison is the validated batch on the
> left; the animation is clearly labelled illustrative.

### Bonus visuals already in `results/`
`evaluation_comparison.png`, `training_reward.png`, `topology.png`,
`congestion_heatmap.png`, `generalisation.png` (GNN transfers to an unseen topology;
fixed-input MLP agents cannot even run on it). Good slides if you want stills.

---

## Part 2 — Hardware Demo (multiple devices)

The hardware stack has **two parts**, both working:

* **Control plane** (SDN-style): each device runs a lightweight UDP *probe server*; a
  *controller* on one laptop probes the links (RTT + loss, like `ping`), feeds those real
  measurements to the trained agent, and the agent outputs the **next-hop decision** —
  shown live next to Dijkstra's choice (`routing_controller.py`, `dashboard.py`).
* **Data plane** (real packet forwarding): each device also runs a *forwarder* daemon.
  The controller computes an agent-guided path, hands a real payload to the first hop, and
  the forwarders move the bytes hop-by-hop to the destination. If a node/link dies, the
  forwarder sends a NACK, the controller **reroutes around the failure, and the data still
  arrives** (`forwarder.py`, `data_plane.py`).

**Try the data plane on one laptop first — no hardware needed:**
```bash
.venv/bin/python -m network_rl.real_network.virtual_demo --src 0 --dst 9
```
It launches one forwarder per node on localhost, delivers a real message, then **kills a
relay node and shows the controller reroute and re-deliver** — with a payload integrity
check. This is the single most convincing thing to show: *you watch real bytes survive a
node failure.*

### Equipment (you said: laptops / phones / Raspberry Pis)
* 1 device = **controller** (runs the dashboard + the agent).
* 2–3 devices = **nodes** (run the probe server). Phones can act as nodes via Termux
  (`pkg install python`) or just use Pis/laptops. All on the **same Wi-Fi/switch**.

### Steps (full version in `setup_real_network.md`)

1. **Static IPs** on the `192.168.10.0/24` subnet (e.g. controller `.1`, nodes `.2`, `.3`, `.4`).
   Verify with `ping` between every pair first.
2. **On each node**, start the probe server *and* the forwarder (different ports):
   ```bash
   python network_rl/real_network/probe_server.py --host 0.0.0.0 --port 9999 --node-id 1
   python network_rl/real_network/forwarder.py     --label 1 --host 0.0.0.0 --port 9000
   ```
3. **Control-plane view** — launch the live decision dashboard on the controller:
   ```bash
   python network_rl/real_network/dashboard.py \
       --model models/dqn_trained.pth \
       --node-map 0:192.168.10.1 1:192.168.10.2 2:192.168.10.3 3:192.168.10.4 \
       --source 0 --dest 2 --interval 3
   ```
   Generate congestion with `iperf3` (`iperf3 -s` on a node, `iperf3 -c <ip> -t 30 -b 10M`
   from the controller) or unplug a node, and watch the panel flip **AGREE → DIVERGE**: the
   agent steers around the degraded link while Dijkstra still points at it.
4. **Data-plane moment (the highlight)** — actually send data across the devices and kill a
   node. The same logic as `virtual_demo.py`, but pointed at real IPs (the forwarders all
   listen on port 9000; pass the node map):
   ```python
   # send_real.py — run on the controller
   from network_rl.real_network.real_env_adapter import RealEnvAdapter
   from network_rl.real_network.data_plane import DataPlaneController, load_agent
   from network_rl.env.network_env import TOPOLOGY_EDGES
   node_map = {0:("192.168.10.1",9000), 1:("192.168.10.2",9000),
               2:("192.168.10.3",9000), 3:("192.168.10.4",9000)}
   adapter = RealEnvAdapter(node_map, [(u,v) for (u,v,_,_) in TOPOLOGY_EDGES])
   agent   = load_agent("models/gnn_trained.pth", kind="gnn")   # the strongest agent
   ctrl = DataPlaneController(agent, adapter, node_map, kind="gnn",
                              reply_host="192.168.10.1", reply_port=9100)
   print(ctrl.send(0, 3, b"Hello across real devices!"))
   ```
   Run it once (data arrives), power off / unplug a relay node, run it again — the controller
   reroutes and the data still arrives. That is the robustness claim, made physical.

> Two fixes landed today that make this demo actually work: (a) loaded agents were secretly
> routing **randomly** (an empty-replay-buffer bug) — now fixed; (b) the real-network adapter
> produced the wrong observation size (19 vs 70 dims) — now fixed. Before these, the
> hardware path was not running the trained policy at all.

**Note on the agent:** the **data plane** routes with the **GNN-DQN by default** (the
strongest agent) — the controller builds a live graph from probe metrics and the GNN routes
over it (`--agent gnn`); `dqn`/`rainbow` are also selectable. The older *control-plane*
decision dashboard (`dashboard.py`) still uses the flat-observation DQN; porting that view to
the GNN is a minor follow-up.

---

## Part 3 — How Far Is This From Real Usage?

Be confident but honest — this is a strong **research prototype / proof-of-concept**, not a
production router. Roughly **TRL 4** (validated in simulation; both control plane *and* a
real packet-forwarding data plane demonstrated on real/virtual nodes).

### What's genuinely solid
* A complete, research-grade simulation: M/M/1 queuing delay, AR(1) + bursty congestion,
  Markov link failures, 7-way algorithm comparison with multi-seed training.
* A learned policy (GNN-DQN) that **beats classical shortest-path on delay while matching/
  beating it on delivery**, and that **generalises to an unseen topology** — a real result.
* An end-to-end bridge to real devices: probe → measure → agent decides → **forwarders carry
  real bytes**, with automatic **reroute around a failed node** (verified by the virtual demo).

### The gaps between this and a deployable router
1. **Application-layer forwarding, not kernel/line-rate.** The data plane forwards payloads in
   user space over TCP using source routing — perfect for a demo, but production routing
   pushes decisions into the kernel/ASIC: Linux `ip route` / `nftables` policy routing, or an
   OpenFlow/P4 switch driven by an SDN controller (Ryu/ONOS). That integration is the next step.
2. **Centralised control.** One controller computes every route (SDN-style). Production needs
   distributed agents or a highly-available controller — no single point of failure.
3. **Fixed, hand-mapped topology.** 10 nodes are hardcoded and IPs are mapped by hand. Real
   usage needs automatic **topology/neighbour discovery** (LLDP-style) and dynamic node
   join/leave. The GNN helps here (it's topology-agnostic); the MLP agents do not.
4. **Sim-to-real gap.** The agent is trained in simulation; the RTT→congestion mapping in the
   adapter is a heuristic calibration. Real deployment needs online fine-tuning on measured
   traffic and proper calibration per link technology.
5. **Scale & state.** The flat observation is tied to 17 edges; it won't transfer to a
   different network without retraining (again, the GNN is the path forward). Large networks
   need hierarchical or per-region agents.
6. **Production hardening.** Convergence guarantees / loop-freedom under partial failure,
   security of the control channel, sub-millisecond inference at line rate, and standards
   interop (BGP/OSPF coexistence) are all open.

### Honest one-liner for the Q&A
> "It's a proof-of-concept that shows a graph-neural-network policy can out-route classical
> shortest-path under dynamic failures — in simulation, and as a working control + data plane
> that forwards real bytes and reroutes around a dead node on real hardware. The path to
> production is kernel/line-rate forwarding, topology auto-discovery, distributed control, and
> on-network fine-tuning — none blocked by the core idea."

### Sensible next steps (in priority order)
1. **Kernel forwarding integration** (push the chosen routes into `ip route` / OpenFlow)
   so forwarding happens at line rate instead of user-space TCP.
2. **Topology auto-discovery** so nodes can join/leave without hand-mapping IPs.
3. **On-network fine-tuning** to close the sim-to-real gap (the agent is trained in sim).
4. **Distributed / HA control** to remove the single-controller bottleneck.

*(Done already: the GNN — the best, topology-agnostic agent — now drives the live data
plane, routing over a graph built from real probe metrics.)*
