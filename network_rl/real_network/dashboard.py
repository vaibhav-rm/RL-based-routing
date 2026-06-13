"""
CLI Live Dashboard for Real Network Mode.

Displays in real time:
  - Measured link latency and loss between probed devices
  - DQN routing decision vs Dijkstra
  - Rolling log of routing decisions

CN concepts:
- The dashboard is analogous to a network NOC (Network Operations Centre)
  display. Link metrics update in a manner similar to SNMP polling cycles.
- The side-by-side comparison (DQN vs Dijkstra) lets you observe when the
  DQN diverges — typically under high-failure / high-congestion scenarios
  where Dijkstra's global-optimum assumption breaks down.

Usage:
    python dashboard.py --model models/dqn_trained.pth \
        --node-map 0:192.168.1.1 1:192.168.1.2 2:192.168.1.3 \
        --source 0 --dest 2 --interval 3
"""

import argparse, os, sys, time, threading
from collections import deque
from typing import Dict, List, Deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rich.console import Console
from rich.table   import Table
from rich.live    import Live
from rich.layout  import Layout
from rich.panel   import Panel
from rich.text    import Text

from network_rl.agents.dqn_agent              import DQNAgent
from network_rl.env.network_env               import NetworkRoutingEnv, TOPOLOGY_EDGES, NUM_NODES
from network_rl.real_network.real_env_adapter import RealEnvAdapter
from network_rl.real_network.probe_client     import probe_link, parse_node_map
from network_rl.real_network.routing_controller import routing_step, build_real_graph, load_agent

console = Console()

MAX_LOG_LINES = 20


class DashboardState:
    def __init__(self):
        self.metrics: Dict[str, dict] = {}   # "u-v" → {rtt, loss, congestion}
        self.last_decision: dict = {}
        self.log: Deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.lock = threading.Lock()


state = DashboardState()


def metrics_table() -> Table:
    tbl = Table(title="Link Metrics (Real Measurements)", border_style="blue")
    tbl.add_column("Link",        style="cyan",   no_wrap=True)
    tbl.add_column("RTT (ms)",    justify="right")
    tbl.add_column("Loss %",      justify="right")
    tbl.add_column("Congestion",  justify="right")
    tbl.add_column("Status",      justify="center")

    with state.lock:
        for link_key, m in state.metrics.items():
            rtt   = m.get("rtt_ms",   -1)
            loss  = m.get("loss_pct",  0)
            cong  = m.get("congestion", 0.5)
            if rtt < 0:
                status = "[red]DOWN[/red]"
                rtt_str = "—"
            elif cong > 0.7:
                status = "[yellow]HIGH CONG[/yellow]"
                rtt_str = f"{rtt:.1f}"
            else:
                status = "[green]OK[/green]"
                rtt_str = f"{rtt:.1f}"

            cong_bar = "█" * int(cong * 10) + "░" * (10 - int(cong * 10))
            tbl.add_row(link_key, rtt_str, f"{loss:.1f}", cong_bar, status)
    return tbl


def decision_panel() -> Panel:
    with state.lock:
        d = state.last_decision
    if not d:
        return Panel("Waiting for first probe cycle…", title="Routing Decision")

    dqn  = d.get("dqn_next_hop",      "N/A")
    dijk = d.get("dijkstra_next_hop",  "N/A")
    src  = d.get("source",  "?")
    dst  = d.get("dest",    "?")
    ts   = d.get("timestamp", "")

    agree = "[green]AGREE[/green]" if dqn == dijk else "[yellow]DIVERGE[/yellow]"
    text = (
        f"[bold]Source → Dest:[/bold] {src} → {dst}   [dim]{ts}[/dim]\n\n"
        f"[bold cyan]DQN next-hop:[/bold cyan]      Node {dqn}\n"
        f"[bold orange1]Dijkstra next-hop:[/bold orange1]  Node {dijk}\n\n"
        f"Consensus: {agree}"
    )
    return Panel(text, title="Routing Decision (DQN vs Dijkstra)", border_style="green")


def log_panel() -> Panel:
    with state.lock:
        lines = list(state.log)
    text = "\n".join(lines[-MAX_LOG_LINES:]) or "[dim]No events yet…[/dim]"
    return Panel(text, title="Routing Event Log", border_style="dim")


def make_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="top",    size=len(state.metrics) + 6),
        Layout(name="middle", size=10),
        Layout(name="bottom"),
    )
    layout["top"].update(metrics_table())
    layout["middle"].update(decision_panel())
    layout["bottom"].update(log_panel())
    return layout


def probe_and_decide(agent, adapter, node_map, source, dest, probe_count):
    """Background thread: probe neighbours, update state, make routing decision."""
    env_tmp = NetworkRoutingEnv()
    env_tmp.reset()
    neighbor_ids = sorted(env_tmp.G.neighbors(source))

    for nid in neighbor_ids:
        if nid not in node_map:
            continue
        host, port = node_map[nid]
        try:
            m = probe_link(host, port, count=probe_count, timeout=1.0, interval=0.05)
        except Exception as e:
            m = {"rtt_ms": -1, "loss_pct": 100.0, "sent": 0, "received": 0}
        cong = adapter._edge_congestion(source, nid) if hasattr(adapter, "_edge_congestion") else 0.5
        adapter.set_link_metrics(source, nid, m["rtt_ms"], m["loss_pct"])
        link_key = f"{source}-{nid}"
        m["congestion"] = adapter._edge_congestion(source, nid)
        with state.lock:
            state.metrics[link_key] = m

    result = routing_step(agent, adapter, source, dest)
    with state.lock:
        state.last_decision = result
        log_entry = (
            f"[{result['timestamp']}] {source}→{dest}  "
            f"DQN:{result['dqn_next_hop']}  "
            f"Dijk:{result['dijkstra_next_hop']}"
        )
        state.log.append(log_entry)


def main():
    parser = argparse.ArgumentParser(description="Real-network routing dashboard")
    parser.add_argument("--model",      default="models/dqn_trained.pth")
    parser.add_argument("--node-map",   nargs="+", default=[],
                        help="simId:ip pairs e.g. 0:192.168.1.1")
    parser.add_argument("--source",     type=int, default=0)
    parser.add_argument("--dest",       type=int, default=2)
    parser.add_argument("--interval",   type=float, default=3.0)
    parser.add_argument("--probe-count", type=int, default=5)
    parser.add_argument("--probe-port",  type=int, default=9999)
    args = parser.parse_args()

    node_map = parse_node_map(args.node_map, default_port=args.probe_port)

    if not node_map:
        console.print("[yellow]No --node-map provided. Running in demo mode "
                      "(localhost, two probe servers on ports 9991/9992).[/yellow]")
        node_map = {1: ("127.0.0.1", 9991), 2: ("127.0.0.1", 9992)}

    edge_order = [(u, v) for (u, v, _, __) in TOPOLOGY_EDGES]
    adapter = RealEnvAdapter(node_map, edge_order, port=args.probe_port)
    agent   = load_agent(args.model)

    console.print(f"[bold green]Dashboard starting[/bold green]  "
                  f"source={args.source} dest={args.dest}  "
                  f"interval={args.interval}s")

    with Live(make_layout(), refresh_per_second=2, console=console) as live:
        while True:
            t = threading.Thread(
                target=probe_and_decide,
                args=(agent, adapter, node_map,
                      args.source, args.dest,
                      args.probe_count),
                daemon=True,
            )
            t.start()
            t.join(timeout=args.interval * 2)
            live.update(make_layout())
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
