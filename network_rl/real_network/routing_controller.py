"""
Real-Network Routing Controller.

Loads the trained DQN from disk and, given real measured link metrics,
recommends the optimal next-hop for each routing decision.

CN concepts:
- Acts as the control plane (like an SDN controller), while the actual
  forwarding plane is the OS routing table or iptables policy routing.
- The DQN inference here replaces the RIB (Routing Information Base)
  computation done by BGP/OSPF in production routers.
- Dijkstra is also computed in parallel for comparison — the controller
  logs both choices so you can see when they diverge under failure conditions.

Usage:
    python routing_controller.py --model models/dqn_trained.pth \
        --node-map 0:192.168.1.1 1:192.168.1.2 2:192.168.1.3 \
        --source 0 --dest 2
"""

import argparse, os, sys, time, json
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from network_rl.agents.dqn_agent import DQNAgent
from network_rl.env.network_env  import NetworkRoutingEnv, TOPOLOGY_EDGES, NUM_NODES
from network_rl.baselines.dijkstra import dijkstra_next_hop
from network_rl.real_network.real_env_adapter import RealEnvAdapter
from network_rl.real_network.probe_client     import probe_all, parse_node_map


def load_agent(model_path: str) -> DQNAgent:
    from network_rl.env.network_env import OBS_DIM, NetworkRoutingEnv
    state_dim  = OBS_DIM
    _tmp_env   = NetworkRoutingEnv()
    action_dim = _tmp_env.action_space.n  # derived from actual topology max-degree
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent.load(model_path)
    agent.epsilon = 0.0
    return agent


def build_real_graph(adapter: RealEnvAdapter) -> nx.Graph:
    """Build a weighted graph from current real metrics for Dijkstra comparison."""
    G = nx.Graph()
    G.add_nodes_from(range(NUM_NODES))
    for (u, v, _, base_delay) in TOPOLOGY_EDGES:
        cong = adapter._edge_congestion(u, v)
        effective = base_delay * (1.0 + 3.0 * cong)
        G.add_edge(u, v, weight=effective)
    return G


def routing_step(
    agent: DQNAgent,
    adapter: RealEnvAdapter,
    source: int,
    dest: int,
) -> dict:
    """Single routing decision: returns DQN next-hop and Dijkstra next-hop."""
    env_for_neighbors = NetworkRoutingEnv()
    env_for_neighbors.reset()

    state = adapter.build_state(source, dest)
    neighbors = sorted(env_for_neighbors.G.neighbors(source))
    valid_actions = list(range(len(neighbors)))

    dqn_action = agent.select_action(state, valid_actions)
    dqn_nexthop = neighbors[dqn_action] if dqn_action < len(neighbors) else None

    graph = build_real_graph(adapter)
    dijk_nexthop = dijkstra_next_hop(graph, source, dest)

    return {
        "timestamp":      time.strftime("%H:%M:%S"),
        "source":         source,
        "dest":           dest,
        "dqn_next_hop":   dqn_nexthop,
        "dijkstra_next_hop": dijk_nexthop,
        "neighbors":      neighbors,
        "congestion_summary": adapter.congestion_summary(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="models/dqn_trained.pth")
    parser.add_argument("--node-map", nargs="+",
                        help="simNodeId:ip pairs e.g. 0:192.168.1.1 1:192.168.1.2",
                        default=[])
    parser.add_argument("--source",  type=int, default=0)
    parser.add_argument("--dest",    type=int, default=9)
    parser.add_argument("--probe-count",  type=int, default=5)
    parser.add_argument("--probe-port",   type=int, default=9999)
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between probing cycles")
    args = parser.parse_args()

    # Parse node map {sim_id: (ip, port)}
    node_map = parse_node_map(args.node_map, default_port=args.probe_port)

    from network_rl.env.network_env import TOPOLOGY_EDGES as _TE
    edge_order = [(u, v) for (u, v, _, __) in _TE]

    adapter = RealEnvAdapter(node_map, edge_order, port=args.probe_port)
    agent   = load_agent(args.model)

    print(f"[Controller] DQN loaded. Routing {args.source} → {args.dest}")
    print(f"[Controller] Node map: {node_map}")

    while True:
        # Probe all neighbours of source that we have a mapping for.
        if node_map:
            env_tmp = NetworkRoutingEnv()
            env_tmp.reset()
            neighbor_ids = sorted(env_tmp.G.neighbors(args.source))
            probe_nodes = [nid for nid in neighbor_ids if nid in node_map]
            targets = [node_map[nid] for nid in probe_nodes]

            if targets:
                probe_results = probe_all(targets, count=args.probe_count)
                for nid, stats in zip(probe_nodes, probe_results.values()):
                    adapter.set_link_metrics(
                        args.source, nid,
                        stats["rtt_ms"], stats["loss_pct"]
                    )

        result = routing_step(agent, adapter, args.source, args.dest)
        print(json.dumps({
            k: v for k, v in result.items() if k != "congestion_summary"
        }, indent=2))
        print(result["congestion_summary"])
        print("-" * 50)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
