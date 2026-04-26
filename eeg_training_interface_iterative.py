"""Iterative-saving entrypoint for the NeuroFocus EEG training collector.

Run this instead of eeg_training_interface.py when collecting training data:

    python eeg_training_interface_iterative.py [same arguments]

It uses eeg_training_interface_robust.py, but drains and flushes queued UDP
samples before manual stage changes and shutdown. It also supports IPv6 loopback
binding, which matters when a sender uses localhost and resolves it to ::1.
"""

import socket

import eeg_training_interface_robust as robust


def socket_family_for_host(host):
    return socket.AF_INET6 if ":" in str(host) else socket.AF_INET


class LoopbackAwareUDPReader(robust.UDPReader):
    def run(self):
        family = socket_family_for_host(self.bind_host)
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as sock:
                sock.bind((self.bind_host, self.port))
                sock.settimeout(0.5)
                self.output_queue.put({
                    "type": "status",
                    "message": f"UDP listening on {self.bind_host}:{self.port} ({'IPv6' if family == socket.AF_INET6 else 'IPv4'})",
                })
                while not self.stop_event.is_set():
                    try:
                        data, address = sock.recvfrom(65535)
                    except socket.timeout:
                        continue
                    except OSError as exc:
                        self.output_queue.put({"type": "error", "message": str(exc)})
                        break
                    row = robust.extract_openbci_bands(data)
                    row["source_host"] = address[0]
                    row["source_port"] = address[1]
                    self.output_queue.put({"type": "packet", "row": row})
        except Exception as exc:
            self.output_queue.put({"type": "error", "message": f"UDP reader failed: {exc}"})


class IterativeSavingTrainingInterface(robust.TrainingInterface):
    def _flush_pending_samples(self):
        """Write queued samples before changing state or closing files."""
        if not hasattr(self, "raw_file"):
            return

        self.drain_samples()

        for handle_name in ("raw_file", "features_file", "diag_file"):
            handle = getattr(self, handle_name, None)
            if handle and not handle.closed:
                handle.flush()

    def next_stage(self):
        self._flush_pending_samples()
        super().next_stage()

    def on_close(self):
        if getattr(self, "_closing", False):
            return
        self._closing = True
        self._flush_pending_samples()
        super().on_close()


# Patch the robust module before main() creates the Tk app.
robust.UDPReader = LoopbackAwareUDPReader
robust.TrainingInterface = IterativeSavingTrainingInterface


def main():
    robust.main()


if __name__ == "__main__":
    main()
