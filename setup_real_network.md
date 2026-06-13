# Real Network Demo — Setup Guide

This guide walks you through running the DQN routing controller on a real LAN
(3–4 machines: laptops, Raspberry Pis, or VMs with virtual interfaces).

---

## Minimum Hardware

| Role | Count | Examples |
|------|-------|---------|
| Routing Controller | 1 | Laptop running the dashboard + DQN |
| Network Nodes | 2–3 | Additional laptops / Raspberry Pi 4 / VMs |

All devices must be on the **same Layer-2 segment** (same Wi-Fi AP or switch).

---

## Step 1 — Assign Static IPs

On each device, assign a static IP in the 192.168.10.0/24 subnet.

### Linux / Raspberry Pi

```bash
# Edit /etc/network/interfaces  OR use nmcli
sudo ip addr add 192.168.10.1/24 dev eth0   # Node 0 (controller)
sudo ip addr add 192.168.10.2/24 dev eth0   # Node 1
sudo ip addr add 192.168.10.3/24 dev eth0   # Node 2
sudo ip addr add 192.168.10.4/24 dev eth0   # Node 3
```

### Windows (PowerShell — admin)

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.10.2 `
    -PrefixLength 24 -DefaultGateway 192.168.10.1
```

### macOS

```bash
sudo ifconfig en0 192.168.10.3 netmask 255.255.255.0
```

Verify with `ping 192.168.10.1` from each device before proceeding.

---

## Step 2 — Install Dependencies

On **every** node:

```bash
pip install -r requirements.txt
```

Minimum packages needed on probe nodes (not running training):
```bash
pip install numpy rich
```

---

## Step 3 — Start the Probe Server on Each Node

On **Node 1** (192.168.10.2):
```bash
python network_rl/real_network/probe_server.py --host 0.0.0.0 --port 9999 --node-id 1
```

On **Node 2** (192.168.10.3):
```bash
python network_rl/real_network/probe_server.py --host 0.0.0.0 --port 9999 --node-id 2
```

On **Node 3** (192.168.10.4):
```bash
python network_rl/real_network/probe_server.py --host 0.0.0.0 --port 9999 --node-id 3
```

Leave these running in the background (`tmux` or `screen` recommended).

---

## Step 4 — Verify Probing Works

From the controller (Node 0 — 192.168.10.1), test the probe client:

```bash
python network_rl/real_network/probe_client.py \
    --targets 192.168.10.2:9999 192.168.10.3:9999 192.168.10.4:9999 \
    --count 20
```

Expected output:
```json
{
  "192.168.10.2:9999": {"rtt_ms": 1.3, "loss_pct": 0.0, "sent": 20, "received": 20},
  "192.168.10.3:9999": {"rtt_ms": 2.1, "loss_pct": 5.0, "sent": 20, "received": 19},
  ...
}
```

---

## Step 5 — Train the DQN (on Controller)

```bash
python train.py
```

This produces `models/dqn_trained.pth` in about 3–8 minutes on CPU.

---

## Step 6 — Launch the DQN Routing Controller

```bash
python network_rl/real_network/routing_controller.py \
    --model models/dqn_trained.pth \
    --node-map 0:192.168.10.1 1:192.168.10.2 2:192.168.10.3 3:192.168.10.4 \
    --source 0 --dest 2 \
    --probe-count 10 --interval 5
```

The controller prints JSON routing decisions every 5 seconds.

---

## Step 7 — Launch the CLI Dashboard

```bash
python network_rl/real_network/dashboard.py \
    --model models/dqn_trained.pth \
    --node-map 0:192.168.10.1 1:192.168.10.2 2:192.168.10.3 3:192.168.10.4 \
    --source 0 --dest 2 \
    --interval 3
```

The dashboard shows:
- Top panel: live link latency, loss, and congestion bar
- Middle panel: DQN vs Dijkstra next-hop decision
- Bottom panel: rolling event log

---

## Step 8 — Generate Test Traffic with iperf3

On **Node 2** (server):
```bash
iperf3 -s -p 5201
```

On **Node 0** (controller, sending traffic):
```bash
iperf3 -c 192.168.10.3 -p 5201 -t 30 -b 10M
```

This creates realistic background traffic that raises congestion on the 0→2 link,
causing the DQN to reroute via Node 1 (if latency increases enough).

---

## Data Plane — Actually Forwarding Real Bytes (with reroute on failure)

The steps above are the **control plane** (the agent *decides* routes). To actually
**move data** across the nodes and show it surviving a failure, use the data plane.

### Single laptop (no hardware) — start here

```bash
.venv/bin/python -m network_rl.real_network.virtual_demo --src 0 --dst 9
```

This launches one forwarder per node on localhost, delivers a real message end-to-end,
then **kills a relay node and shows the controller reroute and re-deliver** the data —
with a payload integrity check. It is the most convincing single command in the project.

### On real devices

1. On **every node**, also run a forwarder (alongside the probe server):
   ```bash
   python network_rl/real_network/forwarder.py --label 1 --host 0.0.0.0 --port 9000
   ```
2. On the **controller**, drive a transfer with `DataPlaneController` (see the snippet in
   `PRESENTATION_GUIDE.md`, Part 2, step 4): it computes an agent-guided path, source-routes
   the payload through the forwarders, and reroutes on any `link_down` NACK or timeout.
3. Power off / unplug a relay node and resend — the data still arrives via a new route.

How it works: the controller embeds the full route (`[label, ip, port]` hops) in each
packet header; forwarders are dumb and topology-agnostic (pure source routing). A node that
can't reach its next hop NACKs the controller, which marks that edge failed, recomputes the
agent's best path over the surviving links, and resends.

---

## VM / Single-Machine Demo (Virtual Interfaces)

If you don't have multiple physical machines, use Linux network namespaces:

```bash
# Create 3 namespaces simulating 3 nodes
sudo ip netns add node0
sudo ip netns add node1
sudo ip netns add node2

# Create virtual ethernet pairs
sudo ip link add veth0 type veth peer name veth1
sudo ip link set veth0 netns node0
sudo ip link set veth1 netns node1

# Assign IPs
sudo ip netns exec node0 ip addr add 192.168.10.1/24 dev veth0
sudo ip netns exec node0 ip link set veth0 up
sudo ip netns exec node1 ip addr add 192.168.10.2/24 dev veth1
sudo ip netns exec node1 ip link set veth1 up

# Run probe server inside node1 namespace
sudo ip netns exec node1 python network_rl/real_network/probe_server.py --port 9999 &

# Test from node0
sudo ip netns exec node0 python network_rl/real_network/probe_client.py \
    --targets 192.168.10.2:9999 --count 5
```

---

## Reading the Dashboard Output

| Field | Meaning |
|-------|---------|
| RTT (ms) | Round-trip time measured by UDP probe |
| Loss % | % probes that timed out (link quality) |
| Congestion bar | Normalised [0–1] metric fed to DQN |
| DQN next-hop | Node ID the DQN recommends forwarding to |
| Dijkstra next-hop | Shortest-path recommendation for comparison |
| AGREE / DIVERGE | Whether both algorithms chose the same next-hop |

**DIVERGE** is most interesting: it usually occurs when a heavily loaded
link is numerically shortest but the DQN has learned to avoid it based on
congestion patterns — demonstrating adaptive routing beyond Dijkstra.
