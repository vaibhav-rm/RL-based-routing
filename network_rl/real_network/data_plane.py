"""
Data-Plane Controller — the RL control plane driving real packet forwarding.

This is the piece that turns "the agent *decided* a route" into "the data
*arrived* despite a failure". It:

  1. computes an agent-guided path from source to destination over the links
     currently believed to be up (the trained DQN/Rainbow ranks next hops; the
     controller masks failed and already-visited links so a single down link
     never strands the packet);
  2. resolves that node path to concrete [label, ip, port] hops and hands the
     payload to the first forwarder, which source-routes it to the destination;
  3. listens for the delivery ACK. If a forwarder reports a broken edge
     (link_down NACK) or the transfer times out, the controller marks the edge
     failed, recomputes a route around it, and resends — up to a retry budget.

The result is RL-guided, failure-resilient delivery of real bytes across real
(or virtual) nodes — the demonstrable "robustness" of the approach.

See virtual_demo.py for a single-laptop end-to-end run (no hardware needed).
"""

import json, os, socket, sys, threading, time, uuid
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from network_rl.env.network_env     import NUM_NODES, OBS_DIM, NetworkRoutingEnv
from network_rl.agents.dqn_agent    import DQNAgent
from network_rl.agents.rainbow_agent import RainbowAgent
from network_rl.agents.gnn_agent    import GNNAgent
from network_rl.real_network.real_env_adapter import RealEnvAdapter
from network_rl.real_network.forwarder        import send_message


def load_agent(model_path: str, kind: str = "gnn"):
    """
    Load a trained agent for live routing. `kind` ∈ {"gnn", "dqn", "rainbow"}.
    The GNN is the strongest agent and is topology-agnostic, so it is the default.
    """
    env = NetworkRoutingEnv()
    if kind == "gnn":
        agent = GNNAgent(graph=env.G, edge_list=env.edge_list)
    elif kind == "rainbow":
        agent = RainbowAgent(state_dim=OBS_DIM, action_dim=env.action_space.n)
    elif kind == "dqn":
        agent = DQNAgent(state_dim=OBS_DIM, action_dim=env.action_space.n)
    else:
        raise ValueError(f"unknown agent kind '{kind}' (expected gnn/dqn/rainbow)")
    agent.load(model_path)
    agent.epsilon = 0.0          # greedy; net is in eval mode after load()
    return agent


class DataPlaneController:
    """
    node_map: {node_id: (ip, data_port)} — where each node's forwarder listens.
    """

    def __init__(self, agent, adapter: RealEnvAdapter,
                 node_map: Dict[int, Tuple[str, int]], kind: str = "gnn",
                 reply_host: str = "127.0.0.1", reply_port: int = 9100,
                 max_hops: int = NUM_NODES * 2):
        self.agent = agent
        self.kind = kind
        self.adapter = adapter
        self.node_map = node_map
        # A NetworkRoutingEnv holds the live graph state: edge features are synced
        # from real measurements before each route computation. The GNN reads this
        # graph directly; the flat-obs agents read adapter.build_state().
        self.env = NetworkRoutingEnv(use_mm1=True)
        self.graph = self.env.G
        self.failed_edges = set()           # {frozenset({u, v})}
        self.reply_host = reply_host
        self.reply_port = reply_port
        self.max_hops = max_hops
        self._status: Dict[str, dict] = {}
        self._status_lock = threading.Lock()
        self._start_status_listener()

    def _sync_graph(self):
        """Reflect current measurements and known failures into the graph features."""
        for (u, v) in self.env.edge_list:
            e = self.env.G.edges[u, v]
            e["active"]     = frozenset({u, v}) not in self.failed_edges
            e["congestion"] = self.adapter._edge_congestion(u, v)
            m = self.adapter._metrics.get(frozenset({u, v}))
            if m is not None:
                e["loss_rate"] = float(min(m.get("loss_pct", 0.0) / 100.0, 0.5))

    # ── agent-guided route computation ──────────────────────────────────────────
    def compute_route(self, src: int, dst: int) -> Optional[List[int]]:
        """Greedy agent roll-out over up + unvisited links. None if no route found."""
        self._sync_graph()
        cur, path, visited = src, [src], {src}
        for _ in range(self.max_hops):
            if cur == dst:
                return path
            neighbors = sorted(self.graph.neighbors(cur))
            valid = [i for i, nb in enumerate(neighbors)
                     if frozenset({cur, nb}) not in self.failed_edges and nb not in visited]
            if not valid:
                return None                 # dead end on the surviving graph
            if self.kind == "gnn":
                order, _ = self.agent.rank_actions(self.env, cur, dst)
                best = next((i for i in order if i in valid), valid[0])
            else:
                state = self.adapter.build_state(cur, dst)
                best = self.agent.rank_actions(state, valid)[0]
            cur = neighbors[best]
            path.append(cur)
            visited.add(cur)
        return path if path[-1] == dst else None

    def _resolve(self, node_path: List[int]) -> List[list]:
        return [[nid, self.node_map[nid][0], int(self.node_map[nid][1])]
                for nid in node_path]

    # ── status (ACK / NACK) listener ─────────────────────────────────────────────
    def _start_status_listener(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.reply_host, self.reply_port))
        self._srv.listen(16)
        threading.Thread(target=self._status_loop, daemon=True).start()

    def _status_loop(self):
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            with conn:
                data = conn.recv(4096)
            for line in data.decode("utf-8", "ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                with self._status_lock:
                    self._status[msg.get("msg_id", "?")] = msg

    def _wait_status(self, msg_id: str, timeout: float) -> Optional[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._status_lock:
                if msg_id in self._status:
                    return self._status.pop(msg_id)
            time.sleep(0.02)
        return None

    # ── the headline operation: resilient send ───────────────────────────────────
    def send(self, src: int, dst: int, payload: bytes,
             attempts: int = 6, ack_timeout: float = 4.0) -> dict:
        """
        Deliver `payload` from src to dst, rerouting around failures.
        Returns a result dict with the outcome and every route attempted.
        """
        tried = []
        for attempt in range(1, attempts + 1):
            route = self.compute_route(src, dst)
            if route is None:
                return {"delivered": False, "reason": "no route over surviving links",
                        "attempts": tried}
            tried.append(route)
            msg_id = uuid.uuid4().hex[:8]
            hops = self._resolve(route)
            header = {"msg_id": msg_id, "hop": 0, "route": hops,
                      "reply_to": [self.reply_host, self.reply_port]}
            first = hops[0]
            print(f"  attempt {attempt}: route {' → '.join(str(n) for n in route)}  (msg {msg_id})")
            try:
                send_message(first[1], int(first[2]), header, payload)
            except OSError as e:
                # The source node's own forwarder is unreachable — mark its first
                # edge failed and try another first hop.
                self.failed_edges.add(frozenset({route[0], route[1]}))
                print(f"    first hop {route[0]}→{route[1]} unreachable ({e}); rerouting")
                continue

            status = self._wait_status(msg_id, ack_timeout)
            if status is None:
                print("    no ACK/NACK (timeout); rerouting")
                # Unknown which hop failed; drop the first edge and retry.
                self.failed_edges.add(frozenset({route[0], route[1]}))
                continue
            if status.get("status") == "delivered":
                return {"delivered": True, "route": route, "attempts": tried,
                        "msg_id": msg_id, "bytes": status.get("bytes")}
            if status.get("status") == "link_down":
                frm, to = int(status["frm"]), int(status["to"])
                self.failed_edges.add(frozenset({frm, to}))
                print(f"    NACK: link {frm}→{to} down; rerouting around it")
                continue

        return {"delivered": False, "reason": "retry budget exhausted", "attempts": tried}
