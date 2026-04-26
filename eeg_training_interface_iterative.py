"""Iterative-saving entrypoint for the NeuroFocus EEG training collector.

Run this instead of eeg_training_interface.py when collecting training data:

    python eeg_training_interface_iterative.py [same arguments]

It uses eeg_training_interface_robust.py, but drains and flushes queued UDP
samples before manual stage changes and shutdown. This prevents samples from
being lost when a section is skipped.
"""

import eeg_training_interface_robust as robust


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
robust.TrainingInterface = IterativeSavingTrainingInterface


def main():
    robust.main()


if __name__ == "__main__":
    main()
