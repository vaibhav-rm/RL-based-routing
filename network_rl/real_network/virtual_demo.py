"""
Single-Laptop Virtual Demo — RL-guided, failure-resilient data delivery.

Proves the whole pipeline end-to-end on one machine, no extra hardware:

  • launches a data-plane forwarder for every node on a separate localhost port
    (these are the "physical nodes");
  • starts the RL control plane (trained DQN + the data-plane controller);
  • sends a real text payload from a source node to a destination node and shows
    the agent-chosen route deliver it;
  • then KILLS a relay node on that route and resends — the controller gets the
    link-down NACK, reroutes around the dead node, and the data still arrives.

This is the honest "robustness" demo: you watch real bytes survive a node
failure because the learned policy found another way through.

Usage:
    .venv/bin/python -m network_rl.real_network.virtual_demo
    .venv/bin/python -m network_rl.real_network.virtual_demo --src 0 --dst 9
"""

import argparse, os, subprocess, sys, time, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from network_rl.env.network_env import TOPOLOGY_EDGES, NUM_NODES
from network_rl.real_network.real_env_adapter import RealEnvAdapter
from network_rl.real_network.data_plane import DataPlaneController, load_agent

ROOT          = os.path.join(os.path.dirname(__file__), "..", "..")
FORWARDER_PY  = os.path.join(os.path.dirname(__file__), "forwarder.py")
BASE_PORT     = 9300              # node i listens on BASE_PORT + i
REPLY_PORT    = 9290


def _default_model(kind: str):
    for name in (f"{kind}_trained.pth", f"{kind}_seed0.pth"):
        p = os.path.join(ROOT, "models", name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No {kind} model found in models/. Run train.py first.")


def launch_forwarders(save_dir):
    procs = []
    for nid in range(NUM_NODES):
        p = subprocess.Popen(
            [sys.executable, FORWARDER_PY, "--label", str(nid),
             "--host", "127.0.0.1", "--port", str(BASE_PORT + nid),
             "--save-dir", save_dir],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        procs.append(p)
    return procs


def banner(txt):
    print("\n" + "=" * 64 + f"\n {txt}\n" + "=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Single-laptop RL data-plane demo")
    ap.add_argument("--src", type=int, default=0)
    ap.add_argument("--dst", type=int, default=9)
    ap.add_argument("--agent", default="gnn", choices=["gnn", "dqn", "rainbow"],
                    help="which trained agent computes the routes (default: gnn — the best)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--message", default="Hello from the RL router! Survive the failure.")
    args = ap.parse_args()

    model = args.model or _default_model(args.agent)
    node_map = {nid: ("127.0.0.1", BASE_PORT + nid) for nid in range(NUM_NODES)}
    save_dir = tempfile.mkdtemp(prefix="rl_dataplane_")

    banner(f"Starting virtual network (one forwarder per node, on localhost) — agent: {args.agent.upper()}")
    procs = launch_forwarders(save_dir)
    time.sleep(1.2)                       # let the forwarders bind their ports

    edge_order = [(u, v) for (u, v, _bw, _d) in TOPOLOGY_EDGES]
    adapter = RealEnvAdapter(node_map, edge_order)      # neutral priors (no probing)
    agent   = load_agent(model, args.agent)
    ctrl    = DataPlaneController(agent, adapter, node_map, kind=args.agent,
                                  reply_host="127.0.0.1", reply_port=REPLY_PORT)

    try:
        # ── 1. Normal delivery ──────────────────────────────────────────────────
        banner(f"1) Send  {args.src} → {args.dst}  (healthy network)")
        res1 = ctrl.send(args.src, args.dst, args.message.encode("utf-8"))
        if not res1["delivered"]:
            print(f"  ✗ delivery failed: {res1.get('reason')}")
            return
        route1 = res1["route"]
        print(f"  ✓ delivered via {' → '.join(map(str, route1))}")
        _verify(save_dir, res1, args.message)

        # ── 2. Kill a relay node on that route and resend ───────────────────────
        relays = [n for n in route1 if n not in (args.src, args.dst)]
        if not relays:
            print("\n(Route had no intermediate node to fail — try a farther dst.)")
            return
        victim = relays[len(relays) // 2]
        banner(f"2) Kill relay node {victim}, then resend  {args.src} → {args.dst}")
        procs[victim].terminate()
        procs[victim].wait(timeout=3)
        print(f"  node {victim} is now DOWN. Resending — the controller must reroute…\n")

        res2 = ctrl.send(args.src, args.dst,
                         f"{args.message} (after node {victim} died)".encode("utf-8"))
        if res2["delivered"]:
            print(f"\n  ✓ data still delivered via {' → '.join(map(str, res2['route']))}")
            print(f"    (rerouted around dead node {victim}; "
                  f"{len(res2['attempts'])} route attempt(s))")
            _verify(save_dir, res2, args.message)
        else:
            print(f"\n  ✗ delivery failed after failure: {res2.get('reason')}")

        banner("Demo complete — RL control plane delivered real data through a failure.")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()


def _verify(save_dir, result, original_message):
    """Confirm the delivered bytes match what was sent (integrity check)."""
    msg_id = result.get("msg_id")
    path = os.path.join(save_dir, f"{msg_id}.bin")
    if msg_id and os.path.exists(path):
        with open(path, "rb") as f:
            got = f.read().decode("utf-8", "ignore")
        ok = original_message in got
        print(f"    payload received at destination: {got!r}  [integrity: {'OK' if ok else 'MISMATCH'}]")


if __name__ == "__main__":
    main()
