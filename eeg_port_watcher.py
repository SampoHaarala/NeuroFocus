#!/usr/bin/env python3
"""Watch localhost EEG traffic from OpenBCI or other senders.

This is intentionally protocol-agnostic. It can listen for UDP datagrams and TCP
connections on the same port, print raw payloads, and make best-effort guesses for
JSON, CSV/text, and binary float payloads.

Typical use:
    python eeg_port_watcher.py --port 12345
    python eeg_port_watcher.py --udp-only --save samples.ndjson
"""

from __future__ import annotations

import argparse
import csv
import json
import select
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 12345
BUFFER_SIZE = 65535


@dataclass
class Packet:
    protocol: str
    source: str
    payload: bytes
    received_at: float


def safe_ascii(payload: bytes, max_chars: int = 240) -> str:
    """Return printable ASCII preview without crashing on binary data."""
    text = payload.decode("utf-8", errors="replace")
    text = "".join(ch if ch.isprintable() or ch in "\r\n\t" else "." for ch in text)
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def hex_preview(payload: bytes, max_bytes: int = 64) -> str:
    preview = payload[:max_bytes].hex(" ")
    if len(payload) > max_bytes:
        preview += " ..."
    return preview


def try_json(payload: bytes) -> object | None:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def try_csv_rows(payload: bytes) -> list[list[str]] | None:
    try:
        text = payload.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        return None

    if not text or "," not in text:
        return None
    try:
        rows = list(csv.reader(text.splitlines()))
    except csv.Error:
        return None
    return rows or None


def try_float_unpack(payload: bytes) -> dict[str, list[float]]:
    """Best-effort binary float interpretation for unknown OpenBCI payloads."""
    result: dict[str, list[float]] = {}

    if len(payload) >= 4 and len(payload) % 4 == 0:
        count = len(payload) // 4
        result["float32_le"] = list(struct.unpack(f"<{count}f", payload))[:16]
        result["float32_be"] = list(struct.unpack(f">{count}f", payload))[:16]

    if len(payload) >= 8 and len(payload) % 8 == 0:
        count = len(payload) // 8
        result["float64_le"] = list(struct.unpack(f"<{count}d", payload))[:16]
        result["float64_be"] = list(struct.unpack(f">{count}d", payload))[:16]

    return result


def format_packet(packet: Packet) -> str:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(packet.received_at))
    lines = [
        f"\n[{timestamp}] {packet.protocol} from {packet.source} | {len(packet.payload)} bytes",
        f"hex:   {hex_preview(packet.payload)}",
        f"text:  {safe_ascii(packet.payload)}",
    ]

    parsed_json = try_json(packet.payload)
    if parsed_json is not None:
        lines.append("json:  " + json.dumps(parsed_json, indent=2, ensure_ascii=False))

    csv_rows = try_csv_rows(packet.payload)
    if csv_rows is not None:
        lines.append(f"csv:   {csv_rows[:5]}")

    float_guesses = try_float_unpack(packet.payload)
    if float_guesses:
        lines.append("floats best-effort:")
        for name, values in float_guesses.items():
            rounded = [round(v, 6) for v in values]
            lines.append(f"  {name}: {rounded}")

    return "\n".join(lines)


def save_packet(path: Path, packet: Packet) -> None:
    record = {
        "received_at": packet.received_at,
        "protocol": packet.protocol,
        "source": packet.source,
        "size_bytes": len(packet.payload),
        "hex": packet.payload.hex(),
        "text_preview": safe_ascii(packet.payload),
        "json": try_json(packet.payload),
        "csv": try_csv_rows(packet.payload),
        "float_guesses": try_float_unpack(packet.payload),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_udp_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.setblocking(False)
    return sock


def create_tcp_server(host: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    server.setblocking(False)
    return server


def watch(host: str, port: int, udp: bool, tcp: bool, save_path: Path | None) -> None:
    sockets: list[socket.socket] = []
    udp_socket: socket.socket | None = None
    tcp_server: socket.socket | None = None
    clients: dict[socket.socket, str] = {}

    if udp:
        udp_socket = create_udp_socket(host, port)
        sockets.append(udp_socket)
        print(f"Listening for UDP on {host}:{port}")

    if tcp:
        tcp_server = create_tcp_server(host, port)
        sockets.append(tcp_server)
        print(f"Listening for TCP on {host}:{port}")

    print("Press Ctrl+C to stop.")

    try:
        while True:
            readable, _, errored = select.select(sockets, [], sockets, 1.0)

            for sock in errored:
                sockets.remove(sock)
                sock.close()

            for sock in readable:
                if sock is udp_socket:
                    payload, address = sock.recvfrom(BUFFER_SIZE)
                    packet = Packet("UDP", f"{address[0]}:{address[1]}", payload, time.time())
                    print(format_packet(packet), flush=True)
                    if save_path:
                        save_packet(save_path, packet)

                elif sock is tcp_server:
                    client, address = tcp_server.accept()
                    client.setblocking(False)
                    source = f"{address[0]}:{address[1]}"
                    clients[client] = source
                    sockets.append(client)
                    print(f"\nTCP client connected from {source}", flush=True)

                else:
                    payload = sock.recv(BUFFER_SIZE)
                    source = clients.get(sock, "unknown")
                    if not payload:
                        sockets.remove(sock)
                        clients.pop(sock, None)
                        sock.close()
                        print(f"\nTCP client disconnected from {source}", flush=True)
                        continue

                    packet = Packet("TCP", source, payload, time.time())
                    print(format_packet(packet), flush=True)
                    if save_path:
                        save_packet(save_path, packet)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for sock in sockets:
            sock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Watch and inspect EEG data sent to a localhost port.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host/interface to bind. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on. Default: {DEFAULT_PORT}")
    parser.add_argument("--udp-only", action="store_true", help="Only listen for UDP packets.")
    parser.add_argument("--tcp-only", action="store_true", help="Only listen for TCP connections.")
    parser.add_argument("--save", type=Path, help="Optional NDJSON file to save captured packets.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.udp_only and args.tcp_only:
        parser.error("Use either --udp-only or --tcp-only, not both.")

    udp = not args.tcp_only
    tcp = not args.udp_only

    try:
        watch(args.host, args.port, udp=udp, tcp=tcp, save_path=args.save)
    except OSError as exc:
        print(f"Could not bind to {args.host}:{args.port}: {exc}", file=sys.stderr)
        print("Check that OpenBCI is sending to this port, and that no other program is already listening.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
