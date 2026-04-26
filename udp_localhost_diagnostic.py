"""UDP localhost diagnostic for NeuroFocus.

Use this to verify that packets actually arrive at Python before debugging EEG
packet parsing or CSV saving.

Terminal 1:
    python udp_localhost_diagnostic.py --listen --port 12345 --host 127.0.0.1

Terminal 2:
    python udp_localhost_diagnostic.py --send --port 12345 --host 127.0.0.1

Also test IPv6 localhost if your sender uses plain `localhost`:

Terminal 1:
    python udp_localhost_diagnostic.py --listen --port 12345 --host ::1
Terminal 2:
    python udp_localhost_diagnostic.py --send --port 12345 --host ::1
"""

import argparse
import socket
import time


def family_for_host(host: str):
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def listen(host: str, port: int):
    family = family_for_host(host)
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.bind((host, port))
        sock.settimeout(1.0)
        print(f"Listening on udp://{host}:{port}")
        print("Waiting for packets. Press Ctrl+C to stop.")
        received = 0
        while True:
            try:
                data, address = sock.recvfrom(65535)
            except socket.timeout:
                print(f"no packets yet ({received} received)")
                continue
            received += 1
            preview = data[:80]
            print(f"packet #{received} from {address}: {len(data)} bytes | {preview!r}")


def send(host: str, port: int, count: int, interval: float):
    family = family_for_host(host)
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        for i in range(count):
            payload = f"neurofocus_udp_test,{i},{time.time()}".encode("utf-8")
            sock.sendto(payload, (host, port))
            print(f"sent {len(payload)} bytes to udp://{host}:{port}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Test local UDP receive/send paths.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--listen", action="store_true", help="listen for UDP packets")
    mode.add_argument("--send", action="store_true", help="send test UDP packets")
    parser.add_argument("--host", default="127.0.0.1", help="bind/send host, e.g. 127.0.0.1, 0.0.0.0, ::1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    if args.listen:
        listen(args.host, args.port)
    else:
        send(args.host, args.port, args.count, args.interval)


if __name__ == "__main__":
    main()
