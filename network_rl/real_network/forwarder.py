"""
Data-Plane Forwarder — runs on every node in the real-network demo.

This is the *forwarding plane* that complements the RL control plane: where the
controller (data_plane.py) decides the path, the forwarders actually move the
bytes across the network, one hop at a time.

It is deliberately dumb and topology-agnostic — pure source routing:

  • A message carries its own route in the header: an ordered list of
    [label, ip, port] hops plus the index of the hop currently holding it.
  • If this node is the last hop, it has arrived → the payload is delivered and
    a "delivered" ACK is returned to the controller.
  • Otherwise the node opens a TCP connection to the next hop and forwards the
    message unchanged (with the hop index advanced by one).
  • If the next hop cannot be reached (node down / link failed), the forwarder
    sends a "link_down" NACK back to the controller naming the broken edge, so
    the controller can recompute a route around it and resend.

Because the full route travels inside the packet, forwarders need no global
configuration — exactly like an IP source-routing option or an MPLS label stack.

Wire format (framed over TCP):
    [4-byte big-endian header length][UTF-8 JSON header][payload bytes]
    header = {
        "msg_id": str, "hop": int, "payload_len": int,
        "route": [[label, ip, port], ...],
        "reply_to": [ip, port],            # controller's status listener
    }

Usage:
    python forwarder.py --label 1 --host 0.0.0.0 --port 9000
    python forwarder.py --label 2 --host 0.0.0.0 --port 9000 --save-dir /tmp/recv
"""

import argparse, json, os, socket, struct, threading, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [forwarder] %(message)s")

CONNECT_TIMEOUT = 2.0


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a stream socket (or raise on early close)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def read_message(conn: socket.socket):
    """Return (header_dict, payload_bytes) from a framed message."""
    (hlen,) = struct.unpack(">I", _recv_exact(conn, 4))
    header  = json.loads(_recv_exact(conn, hlen).decode("utf-8"))
    payload = _recv_exact(conn, header["payload_len"])
    return header, payload


def encode_message(header: dict, payload: bytes) -> bytes:
    header = dict(header)
    header["payload_len"] = len(payload)
    hb = json.dumps(header).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + payload


def send_message(host: str, port: int, header: dict, payload: bytes,
                 timeout: float = CONNECT_TIMEOUT):
    """Open a short-lived TCP connection and send one framed message."""
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(encode_message(header, payload))


def _send_status(reply_to, status: dict):
    """Best-effort status report (ACK/NACK) back to the controller."""
    if not reply_to:
        return
    try:
        host, port = reply_to
        with socket.create_connection((host, int(port)), timeout=CONNECT_TIMEOUT) as s:
            s.sendall((json.dumps(status) + "\n").encode("utf-8"))
    except OSError:
        pass  # controller not listening / gone — nothing we can do from here


class Forwarder:
    def __init__(self, label, host, port, save_dir=None):
        self.label = label
        self.host = host
        self.port = port
        self.save_dir = save_dir
        self.log = logging.getLogger()

    def handle(self, conn):
        try:
            header, payload = read_message(conn)
        except (ConnectionError, ValueError, struct.error) as e:
            self.log.warning(f"node {self.label}: bad message: {e}")
            return

        route    = header["route"]
        hop      = header["hop"]
        reply_to = header.get("reply_to")
        msg_id   = header.get("msg_id", "?")
        here     = route[hop][0]

        # Arrived at the final hop → deliver.
        if hop >= len(route) - 1:
            text = self._deliver(msg_id, payload)
            self.log.info(f"node {here}: DELIVERED msg {msg_id} ({len(payload)} bytes): {text}")
            _send_status(reply_to, {"msg_id": msg_id, "status": "delivered",
                                    "at": here, "bytes": len(payload)})
            return

        nxt = route[hop + 1]           # [label, ip, port]
        header["hop"] = hop + 1
        try:
            send_message(nxt[1], int(nxt[2]), header, payload)
            self.log.info(f"node {here}: forwarded msg {msg_id} → node {nxt[0]}")
        except OSError as e:
            self.log.warning(f"node {here}: next hop → {nxt[0]} unreachable ({e}); NACK")
            _send_status(reply_to, {"msg_id": msg_id, "status": "link_down",
                                    "frm": here, "to": nxt[0]})

    def _deliver(self, msg_id, payload: bytes) -> str:
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            path = os.path.join(self.save_dir, f"{msg_id}.bin")
            with open(path, "wb") as f:
                f.write(payload)
        try:
            return payload.decode("utf-8")[:80]
        except UnicodeDecodeError:
            return f"<{len(payload)} binary bytes>"

    def serve(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(16)
        self.log.info(f"forwarder listening on {self.host}:{self.port} (TCP)")
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=self._handle_and_close,
                             args=(conn,), daemon=True).start()

    def _handle_and_close(self, conn):
        with conn:
            self.handle(conn)


def main():
    p = argparse.ArgumentParser(description="Data-plane source-routing forwarder")
    p.add_argument("--label", required=True, help="this node's label (e.g. its sim node id)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--save-dir", default="", help="if set, delivered payloads are written here")
    args = p.parse_args()
    Forwarder(args.label, args.host, args.port, args.save_dir or None).serve()


if __name__ == "__main__":
    main()
