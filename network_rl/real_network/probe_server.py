"""
Probe Server — runs on each physical/virtual node in the real network demo.

CN concepts:
- Listens on a UDP port and echoes back any received datagram with a timestamp.
  This mimics ICMP Echo Reply (used by ping/traceroute) but in user space so
  no root privileges are required.
- UDP is preferred over TCP here because we want to measure raw link latency
  without TCP's retransmission / flow-control overhead.

Link emulation (for the single-machine virtual demo):
- --delay-ms adds artificial one-way processing delay before replying, so a
  localhost link can mimic a congested WAN hop.
- --loss-pct drops that fraction of replies, mimicking packet loss / a flaky link.
- --control-file points at a text file containing "<delay_ms> <loss_pct>" that is
  re-read on every packet, so congestion/loss can be injected or cleared live
  (e.g. from a script) without restarting the server.

Usage:
    python probe_server.py [--host 0.0.0.0] [--port 9999] [--node-id 1]
                           [--delay-ms 0] [--loss-pct 0] [--control-file PATH]
"""

import argparse
import os
import random
import socket
import struct
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [probe_server] %(message)s")


def _read_control(control_file: str, default_delay: float, default_loss: float):
    """Return (delay_ms, loss_pct), re-read live from control_file if present."""
    if not control_file:
        return default_delay, default_loss
    try:
        with open(control_file, "r") as f:
            parts = f.read().split()
        delay = float(parts[0]) if len(parts) >= 1 else default_delay
        loss  = float(parts[1]) if len(parts) >= 2 else default_loss
        return delay, loss
    except (OSError, ValueError):
        # File missing or malformed → fall back to the CLI defaults.
        return default_delay, default_loss


def run_server(host: str, port: int, node_id: int,
               delay_ms: float = 0.0, loss_pct: float = 0.0,
               control_file: str = ""):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    extra = ""
    if delay_ms or loss_pct or control_file:
        extra = f" [emulation: delay={delay_ms}ms loss={loss_pct}%"
        extra += f" control={control_file}]" if control_file else "]"
    logging.info(f"Node {node_id}: listening on {host}:{port} (UDP){extra}")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            # Payload: 8-byte client send-timestamp (double, big-endian) + optional padding
            if len(data) < 8:
                continue

            cur_delay, cur_loss = _read_control(control_file, delay_ms, loss_pct)

            # Simulate packet loss by silently dropping the reply.
            if cur_loss > 0 and random.random() * 100.0 < cur_loss:
                continue
            # Simulate link/queuing delay before replying.
            if cur_delay > 0:
                time.sleep(cur_delay / 1000.0)

            client_ts = struct.unpack(">d", data[:8])[0]
            server_ts = time.time()
            # Reply: original client_ts + server receipt time
            reply = struct.pack(">dd", client_ts, server_ts)
            sock.sendto(reply, addr)
        except Exception as e:
            logging.warning(f"Server error: {e}")


def main():
    parser = argparse.ArgumentParser(description="UDP probe echo server")
    parser.add_argument("--host",         default="0.0.0.0")
    parser.add_argument("--port",         type=int,   default=9999)
    parser.add_argument("--node-id",      type=int,   default=0)
    parser.add_argument("--delay-ms",     type=float, default=0.0,
                        help="artificial reply delay in ms (link/queuing emulation)")
    parser.add_argument("--loss-pct",     type=float, default=0.0,
                        help="percent of replies to drop (packet-loss emulation)")
    parser.add_argument("--control-file", default="",
                        help="path to a live-readable '<delay_ms> <loss_pct>' file")
    args = parser.parse_args()
    run_server(args.host, args.port, args.node_id,
               delay_ms=args.delay_ms, loss_pct=args.loss_pct,
               control_file=args.control_file)


if __name__ == "__main__":
    main()
