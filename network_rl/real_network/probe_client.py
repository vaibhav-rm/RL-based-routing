"""
Probe Client — sends probe packets to each neighbour, measures RTT and loss.

CN concepts:
- RTT / 2 ≈ one-way delay (used as the link metric, like OSPF metric).
- Packet loss % is estimated over a window of N probes — mirrors how BGP
  keepalives detect neighbour unreachability.
- Probing interval (default 1 s) is similar to BFD (Bidirectional Forwarding
  Detection) hello intervals used in real routers.

Usage:
    python probe_client.py --targets 192.168.1.2:9999 192.168.1.3:9999 \
                           --count 10 --timeout 0.5
"""

import argparse
import socket
import struct
import time
import json
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [probe_client] %(message)s")

PROBE_PAYLOAD_SIZE = 64  # bytes (matches ICMP default)


def probe_link(
    target_host: str,
    target_port: int,
    count: int = 10,
    timeout: float = 1.0,
    interval: float = 0.1,
) -> Dict:
    """
    Send `count` UDP probe packets to target and return link metrics.

    Returns:
        {
          "host": str, "port": int,
          "rtt_ms": float,        # average RTT in milliseconds
          "jitter_ms": float,     # std-dev of RTT
          "loss_pct": float,      # packet loss percentage [0–100]
          "sent": int, "received": int
        }
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)

    rtts: List[float] = []
    sent = 0
    received = 0

    # Pad payload to PROBE_PAYLOAD_SIZE bytes (like ping -s)
    padding = b"\x00" * (PROBE_PAYLOAD_SIZE - 8)

    for _ in range(count):
        send_ts = time.time()
        payload = struct.pack(">d", send_ts) + padding
        try:
            sock.sendto(payload, (target_host, target_port))
            sent += 1
            data, _ = sock.recvfrom(1024)
            recv_ts = time.time()
            if len(data) >= 16:
                orig_ts, server_ts = struct.unpack(">dd", data[:16])
                rtt_ms = (recv_ts - orig_ts) * 1000.0
                rtts.append(rtt_ms)
                received += 1
        except socket.timeout:
            pass  # count as dropped packet
        except Exception as e:
            logging.debug(f"Probe error to {target_host}:{target_port} — {e}")

        time.sleep(interval)

    sock.close()

    loss_pct = ((sent - received) / sent * 100.0) if sent > 0 else 100.0
    return {
        "host":      target_host,
        "port":      target_port,
        "rtt_ms":    float(round(sum(rtts) / len(rtts), 2)) if rtts else -1.0,
        "jitter_ms": float(round(float(np.std(rtts)), 2)) if len(rtts) > 1 else 0.0,
        "loss_pct":  round(loss_pct, 1),
        "sent":      sent,
        "received":  received,
    }


def probe_all(targets: List[Tuple[str, int]], count: int = 10,
              timeout: float = 1.0) -> Dict[str, Dict]:
    """Probe all neighbours and return a dict keyed by 'host:port'."""
    results = {}
    for host, port in targets:
        key = f"{host}:{port}"
        logging.info(f"Probing {key} ({count} packets)…")
        results[key] = probe_link(host, port, count=count, timeout=timeout)
    return results


def parse_targets(target_strings: List[str]) -> List[Tuple[str, int]]:
    out = []
    for t in target_strings:
        parts = t.rsplit(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) == 2 else 9999
        out.append((host, port))
    return out


def parse_node_map(entries: List[str], default_port: int = 9999) -> Dict[int, Tuple[str, int]]:
    """
    Parse --node-map CLI entries into {sim_node_id: (ip, port)}.

    Accepts either "simId:ip" (port defaults to `default_port`) or
    "simId:ip:port" so several virtual nodes can share one host on
    different ports (used by the single-machine virtual demo).
    """
    node_map: Dict[int, Tuple[str, int]] = {}
    for entry in entries:
        parts = entry.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid --node-map entry '{entry}' (expected simId:ip[:port])")
        sim_id = int(parts[0])
        ip     = parts[1]
        port   = int(parts[2]) if len(parts) >= 3 else default_port
        node_map[sim_id] = (ip, port)
    return node_map


def main():
    import numpy as np  # local import so probe_link() works without numpy if used alone

    parser = argparse.ArgumentParser(description="UDP probe client")
    parser.add_argument("--targets", nargs="+", required=True,
                        help="host:port pairs, e.g. 192.168.1.2:9999")
    parser.add_argument("--count",   type=int,   default=10)
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    targets = parse_targets(args.targets)
    results = probe_all(targets, count=args.count, timeout=args.timeout)
    print(json.dumps(results, indent=2))


# Allow numpy std inside probe_link when called standalone
try:
    import numpy as np
except ImportError:
    import math
    class _FakeNp:
        def std(self, lst):
            if not lst:
                return 0.0
            mean = sum(lst) / len(lst)
            return math.sqrt(sum((x - mean) ** 2 for x in lst) / len(lst))
    np = _FakeNp()


if __name__ == "__main__":
    main()
