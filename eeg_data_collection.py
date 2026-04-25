import argparse
import csv
import math
import queue
import random
import socket
import struct
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import pandas as pd


DEFAULT_BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

READING_TEXT = """
EEG rhythms are patterns of brain activity measured from the scalp. These rhythms
are often grouped into frequency bands. Delta activity is slow and is often linked
to deep sleep. Theta activity is slower than alpha and may appear during drowsy or
internally focused states. Alpha activity is often stronger during relaxed wakeful
states, especially with eyes closed. Beta activity is commonly linked to alertness,
active thinking, and problem solving. Gamma activity is faster and is sometimes
associated with complex information processing.

Read carefully and try to remember the difference between the rhythm bands. This
stage is intended to create focused study-like EEG samples.
""".strip()


@dataclass
class Stage:
    name: str
    label: str | None
    kind: str
    duration_seconds: int
    collect: bool


def iso_now():
    return datetime.now().isoformat(timespec="milliseconds")


def safe_subject_filename(subject_id):
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in subject_id.strip())
    return cleaned or "subject"


def default_features_path(raw_output_csv, subject_id):
    raw_path = Path(raw_output_csv)
    return raw_path.with_name(f"{safe_subject_filename(subject_id)}-features.csv")


def unpack_float32_be_packet(data):
    """Decode a binary packet as big-endian float32 values."""
    usable_size = len(data) - (len(data) % 4)
    if usable_size <= 0:
        return []
    count = usable_size // 4
    return [float(v) for v in struct.unpack(f">{count}f", data[:usable_size])]


def is_valid_band_window(values, tolerance=0.08):
    if len(values) != 5:
        return False
    if not all(math.isfinite(v) for v in values):
        return False
    if not all(0.0 <= v <= 1.0 for v in values):
        return False
    return abs(sum(values) - 1.0) <= tolerance


def choose_band_values(floats, band_count=5, tolerance=0.08):
    """Find the useful normalized five-band window inside decoded packet floats.

    The packet may contain leading garbage/header-like floats. Example bad value:
    1.5e+23. That is not a normalized EEG band value. We scan from the end and
    keep the last five-float window that is finite, inside [0, 1], and sums close
    to 1.0.
    """
    if len(floats) < band_count:
        return None, "too_few_floats"

    for start in range(len(floats) - band_count, -1, -1):
        window = [float(v) for v in floats[start : start + band_count]]
        if is_valid_band_window(window, tolerance=tolerance):
            return window, f"normalized_window_start_{start}"

    # Last-resort fallback: sometimes one garbage/header value lands directly
    # before the final four good values. Use the last five sane values and
    # normalize them so ML still gets five bounded features.
    sane = [float(v) for v in floats if math.isfinite(v) and 0.0 <= v <= 1.0]
    if len(sane) >= band_count and sum(sane[-band_count:]) > 0:
        window = sane[-band_count:]
        total = sum(window)
        return [v / total for v in window], "normalized_last_sane_values"

    return None, "no_valid_normalized_window"


def extract_last_normalized_bands(data, band_names=None, tolerance=0.08):
    """Extract normalized EEG band values from an OpenBCI binary packet."""
    if band_names is None:
        band_names = DEFAULT_BAND_NAMES

    floats = unpack_float32_be_packet(data)
    values, method = choose_band_values(floats, band_count=len(band_names), tolerance=tolerance)
    if values is None:
        print(f"Warning: skipped packet. reason={method}, floats={floats}")
        return None

    row = {
        "packet_time": iso_now(),
        "packet_unix_time": time.time(),
        "packet_size_bytes": len(data),
        "decoded_float_count": len(floats),
        "extraction_method": method,
    }
    for band, value in zip(band_names, values):
        row[band] = value
    return row


def parse_openbci_band_power_line(line):
    """Parse a legacy OpenBCI text band-power line into channel and band values."""
    line = line.strip()
    if not line:
        return None
    try:
        address_part, data_part = line.split("|", 1)
    except ValueError:
        raise ValueError(f"Unable to split line into address and data: {line}")
    address = address_part.strip()
    if not data_part.strip().lower().startswith("data:"):
        raise ValueError(f"OpenBCI band-power line missing 'Data:' section: {line}")
    channel = int(address.split("/")[-1])
    value_text = data_part.split(":", 1)[1].strip()
    if value_text.startswith("(") and value_text.endswith(")"):
        value_text = value_text[1:-1]
    values = [float(v.strip()) for v in value_text.split(",") if v.strip()]
    return channel, values


def openbci_lines_to_dataframe(lines, num_channels=4, band_names=None):
    """Convert legacy OpenBCI band-power text lines into a pandas DataFrame."""
    if band_names is None:
        band_names = DEFAULT_BAND_NAMES
    rows = []
    current_window = {}
    for line in lines:
        parsed = parse_openbci_band_power_line(line)
        if parsed is None:
            continue
        channel, values = parsed
        if len(values) != len(band_names):
            raise ValueError(f"Expected {len(band_names)} band values, got {len(values)}: {line}")
        current_window[channel] = values
        if len(current_window) == num_channels:
            feature_row = {}
            for ch in sorted(current_window):
                for idx, band in enumerate(band_names):
                    feature_row[f"ch{ch}_{band}"] = current_window[ch][idx]
            rows.append(feature_row)
            current_window = {}
    if current_window:
        print("Warning: dropped incomplete final OpenBCI window")
    return pd.DataFrame(rows)


def load_openbci_band_power_log(path, num_channels=4, band_names=None):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return openbci_lines_to_dataframe(lines, num_channels=num_channels, band_names=band_names)


def make_overlapping_windows(df, window_size_seconds=3.0, step_size_seconds=1.5, sample_rate=1.0):
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    window_size = max(1, int(round(window_size_seconds * sample_rate)))
    step_size = max(1, int(round(step_size_seconds * sample_rate)))
    metadata_columns = {
        "packet_time", "packet_unix_time", "packet_size_bytes", "decoded_float_count",
        "extraction_method", "source_host", "source_port", "recorded_time",
        "recorded_unix_time", "subject_id", "session_id", "stage_name", "task_kind",
        "task_detail", "task_answer", "stage_elapsed_seconds", "label",
    }
    feature_columns = [c for c in df.columns if c not in metadata_columns]
    if len(df) < window_size:
        return pd.DataFrame(columns=feature_columns + [
            "window_start_row", "window_end_row", "window_size_samples",
            "window_start_sec", "window_end_sec", "window_midpoint_sec",
        ])
    rows = []
    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start : start + window_size]
        avg = window[feature_columns].mean(axis=0)
        avg["window_start_row"] = start
        avg["window_end_row"] = start + window_size - 1
        avg["window_size_samples"] = window_size
        avg["window_start_sec"] = start / sample_rate
        avg["window_end_sec"] = (start + window_size - 1) / sample_rate
        avg["window_midpoint_sec"] = (avg["window_start_sec"] + avg["window_end_sec"]) / 2.0
        rows.append(avg)
    return pd.DataFrame(rows)


def load_label_schedule(path):
    schedule = pd.read_csv(path)
    required = {"start_seconds", "end_seconds", "label"}
    if not required.issubset(set(schedule.columns)):
        raise ValueError("Label schedule file must contain start_seconds,end_seconds,label columns")
    schedule = schedule.copy()
    schedule["start_seconds"] = schedule["start_seconds"].astype(float)
    schedule["end_seconds"] = schedule["end_seconds"].astype(float)
    schedule["label"] = schedule["label"].astype(str)
    return schedule


def assign_labels_to_windows(window_df, schedule_df, default_label=None, label_column="label"):
    if "window_midpoint_sec" not in window_df.columns:
        raise ValueError("Window DataFrame must contain window_midpoint_sec for automatic labeling")
    labels = []
    for time_sec in window_df["window_midpoint_sec"]:
        matches = schedule_df[(schedule_df["start_seconds"] <= time_sec) & (schedule_df["end_seconds"] > time_sec)]
        labels.append(matches.iloc[0]["label"] if not matches.empty else default_label)
    result = window_df.copy()
    result[label_column] = labels
    return result


def receive_openbci_band_power_udp(port, num_channels=4, band_names=None, timeout=10.0, max_windows=None):
    """Receive binary OpenBCI packets over UDP and return normalized band rows."""
    if band_names is None:
        band_names = DEFAULT_BAND_NAMES
    rows = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("", port))
        if timeout is not None:
            sock.settimeout(timeout)
        print(f"Listening for binary OpenBCI float32_be data on UDP port {port}...")
        while True:
            try:
                data, address = sock.recvfrom(65535)
            except socket.timeout:
                print("Receive timeout reached, stopping capture.")
                break
            row = extract_last_normalized_bands(data, band_names=band_names)
            if row is None:
                continue
            row["source_host"] = address[0]
            row["source_port"] = address[1]
            rows.append(row)
            print("Captured sample " + f"{len(rows)}: " + ", ".join(f"{band}={row[band]:.6f}" for band in band_names), flush=True)
            if max_windows is not None and len(rows) >= max_windows:
                print(f"Captured {len(rows)} samples, stopping.")
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def stream_openbci_band_power_udp(port, num_channels=4, band_names=None, timeout=10.0, max_windows=None):
    if band_names is None:
        band_names = DEFAULT_BAND_NAMES
    emitted = 0
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("", port))
        if timeout is not None:
            sock.settimeout(timeout)
        print(f"Listening for binary OpenBCI float32_be data on UDP port {port}...")
        while True:
            try:
                data, address = sock.recvfrom(65535)
            except socket.timeout:
                print("Receive timeout reached, stopping capture.")
                break
            row = extract_last_normalized_bands(data, band_names=band_names)
            if row is None:
                continue
            row["source_host"] = address[0]
            row["source_port"] = address[1]
            yield row
            emitted += 1
            if max_windows is not None and emitted >= max_windows:
                print(f"Captured {emitted} samples, stopping.")
                return


def parse_band_names(text):
    if not text:
        return DEFAULT_BAND_NAMES
    names = [name.strip() for name in text.split(",") if name.strip()]
    if len(names) != 5:
        raise ValueError("Exactly five band names are required")
    return names


class UDPReader(threading.Thread):
    def __init__(self, port, output_queue, stop_event, band_names):
        super().__init__(daemon=True)
        self.port = port
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.band_names = band_names

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("", self.port))
            sock.settimeout(0.5)
            while not self.stop_event.is_set():
                try:
                    data, address = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError as exc:
                    self.output_queue.put({"type": "error", "message": str(exc)})
                    break
                row = extract_last_normalized_bands(data, band_names=self.band_names)
                if row is None:
                    continue
                row["source_host"] = address[0]
                row["source_port"] = address[1]
                self.output_queue.put({"type": "sample", "row": row})


class TrainingInterface:
    def __init__(self, root, args, band_names):
        self.root = root
        self.args = args
        self.band_names = band_names
        self.subject_id = args.subject_id
        self.session_id = args.session_id or time.strftime("%Y%m%d_%H%M%S")
        self.raw_output_path = Path(args.raw_output_csv)
        self.features_output_path = Path(args.features_output_csv or default_features_path(args.raw_output_csv, self.subject_id))
        self.sample_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.raw_file = None
        self.features_file = None
        self.raw_writer = None
        self.features_writer = None
        self.sample_count = 0
        self.labeled_sample_count = 0
        self.stage_index = -1
        self.stage_started_at = None
        self.current_answer = ""
        self.current_problem = ""
        self.problem_answer = None
        self.math_correct = 0
        self.math_total = 0
        self.relax_items = []
        self.stages = [
            Stage("Prepare", None, "instruction", args.prepare_seconds, False),
            Stage("Focused reading", "focused_reading", "reading", args.focus_seconds, True),
            Stage("Focused arithmetic", "focused_math", "math", args.math_seconds, True),
            Stage("Relaxed scenery", "relaxed", "relax", args.relax_seconds, True),
            Stage("Finished", None, "finished", 0, False),
        ]
        self.root.title("NeuroFocus EEG Training Collector")
        self.root.geometry("980x720")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.open_csvs()
        self.reader = UDPReader(args.recv_port, self.sample_queue, self.stop_event, band_names)
        self.reader.start()
        self.next_stage()
        self.root.after(100, self.tick)

    def build_ui(self):
        self.top_frame = ttk.Frame(self.root, padding=12)
        self.top_frame.pack(fill="x")
        self.title_label = ttk.Label(self.top_frame, text="", font=("Arial", 22, "bold"))
        self.title_label.pack(anchor="w")
        self.status_label = ttk.Label(self.top_frame, text="", font=("Arial", 11))
        self.status_label.pack(anchor="w", pady=(4, 0))
        self.progress = ttk.Progressbar(self.top_frame, maximum=100)
        self.progress.pack(fill="x", pady=(8, 0))
        self.content_frame = ttk.Frame(self.root, padding=12)
        self.content_frame.pack(fill="both", expand=True)
        self.bottom_frame = ttk.Frame(self.root, padding=12)
        self.bottom_frame.pack(fill="x")
        self.sample_label = ttk.Label(self.bottom_frame, text="Samples seen: 0 | Labeled saved: 0")
        self.sample_label.pack(side="left")
        ttk.Button(self.bottom_frame, text="Next stage", command=self.next_stage).pack(side="right", padx=(8, 0))
        ttk.Button(self.bottom_frame, text="Stop", command=self.on_close).pack(side="right")

    def open_writer(self, path, fieldnames):
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists() and path.stat().st_size > 0
        handle = path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            handle.flush()
        return handle, writer

    def open_csvs(self):
        raw_fields = [
            "recorded_time", "recorded_unix_time", "subject_id", "session_id", "stage_name",
            "label", "task_kind", "task_detail", "task_answer", "stage_elapsed_seconds",
            "packet_time", "packet_unix_time", "packet_size_bytes", "decoded_float_count",
            "extraction_method", "source_host", "source_port", *self.band_names,
        ]
        feature_fields = ["subject_id", "session_id", "label", *self.band_names]
        self.raw_file, self.raw_writer = self.open_writer(self.raw_output_path, raw_fields)
        self.features_file, self.features_writer = self.open_writer(self.features_output_path, feature_fields)

    def clear_content(self):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        self.relax_items = []

    def current_stage(self):
        return self.stages[self.stage_index] if 0 <= self.stage_index < len(self.stages) else None

    def next_stage(self):
        self.stage_index += 1
        if self.stage_index >= len(self.stages):
            self.on_close()
            return
        stage = self.stages[self.stage_index]
        self.stage_started_at = time.time()
        self.current_answer = ""
        self.current_problem = ""
        self.problem_answer = None
        self.clear_content()
        self.title_label.config(text=stage.name)
        if stage.kind == "instruction":
            self.show_instruction()
        elif stage.kind == "reading":
            self.show_reading()
        elif stage.kind == "math":
            self.show_math()
        elif stage.kind == "relax":
            self.show_relax()
        elif stage.kind == "finished":
            self.show_finished()

    def show_instruction(self):
        text = (
            f"Connect OpenBCI so it sends UDP packets to port {self.args.recv_port}.\n\n"
            "The raw file contains metadata + features. The features file contains only label + five EEG features.\n\n"
            f"Raw: {self.raw_output_path}\nFeatures: {self.features_output_path}\n\n"
            "Data is only saved during labeled task stages. Preparation and finished screens are not collected."
        )
        ttk.Label(self.content_frame, text=text, wraplength=900, font=("Arial", 16)).pack(anchor="nw")

    def show_reading(self):
        ttk.Label(self.content_frame, text="Read carefully. Label: focused_reading", font=("Arial", 15, "bold"), wraplength=900).pack(anchor="nw", pady=(0, 12))
        text_widget = tk.Text(self.content_frame, wrap="word", font=("Arial", 15), height=18)
        text_widget.insert("1.0", READING_TEXT)
        text_widget.config(state="disabled")
        text_widget.pack(fill="both", expand=True)

    def make_problem(self):
        kind = random.choice(["mul", "two_step", "three_step"])
        if kind == "mul":
            a, b = random.randint(12, 29), random.randint(3, 12)
            self.current_problem, self.problem_answer = f"{a} × {b}", a * b
        elif kind == "two_step":
            a, b, c = random.randint(10, 40), random.randint(2, 9), random.randint(10, 60)
            self.current_problem, self.problem_answer = f"({a} × {b}) - {c}", (a * b) - c
        else:
            a, b, c, d = random.randint(4, 12), random.randint(6, 18), random.randint(20, 90), random.randint(2, 9)
            self.current_problem, self.problem_answer = f"({a} × {b}) + {c} - {d}", (a * b) + c - d

    def show_math(self):
        ttk.Label(self.content_frame, text="Solve arithmetic. Label: focused_math", font=("Arial", 15, "bold"), wraplength=900).pack(anchor="nw", pady=(0, 20))
        self.math_problem_label = ttk.Label(self.content_frame, text="", font=("Arial", 34, "bold"))
        self.math_problem_label.pack(pady=(20, 12))
        self.answer_var = tk.StringVar()
        answer_entry = ttk.Entry(self.content_frame, textvariable=self.answer_var, font=("Arial", 24), justify="center")
        answer_entry.pack(ipady=8)
        answer_entry.bind("<Return>", self.submit_math_answer)
        answer_entry.focus_set()
        self.math_feedback_label = ttk.Label(self.content_frame, text="", font=("Arial", 14))
        self.math_feedback_label.pack(pady=(12, 0))
        self.make_problem()
        self.math_problem_label.config(text=self.current_problem)

    def submit_math_answer(self, event=None):
        text = self.answer_var.get().strip()
        self.current_answer = text
        self.math_total += 1
        try:
            is_correct = int(text) == self.problem_answer
        except ValueError:
            is_correct = False
        if is_correct:
            self.math_correct += 1
            self.math_feedback_label.config(text=f"Correct. Score: {self.math_correct}/{self.math_total}")
        else:
            self.math_feedback_label.config(text=f"Wrong. Correct answer was {self.problem_answer}. Score: {self.math_correct}/{self.math_total}")
        self.answer_var.set("")
        self.make_problem()
        self.math_problem_label.config(text=self.current_problem)

    def show_relax(self):
        ttk.Label(self.content_frame, text="Relax and watch the scenery. Label: relaxed", font=("Arial", 15, "bold"), wraplength=900).pack(anchor="nw", pady=(0, 12))
        self.canvas = tk.Canvas(self.content_frame, width=900, height=430, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.draw_relax_scene()
        self.animate_relax_scene()

    def draw_relax_scene(self):
        self.canvas.delete("all")
        width, height = 900, 430
        self.canvas.create_rectangle(0, 0, width, height, fill="#b7ddff", outline="")
        self.canvas.create_oval(650, 45, 730, 125, fill="#fff2a8", outline="")
        self.canvas.create_rectangle(0, 260, width, height, fill="#8ed081", outline="")
        self.canvas.create_rectangle(0, 330, width, height, fill="#7fc8f8", outline="")
        for x in range(0, width, 80):
            self.canvas.create_arc(x, 250, x + 120, 390, start=0, extent=180, fill="#5fa85a", outline="")
        self.relax_items = []
        for _ in range(6):
            x, y = random.randint(0, width - 80), random.randint(60, 170)
            cloud = self.canvas.create_oval(x, y, x + 80, y + 35, fill="white", outline="")
            self.relax_items.append((cloud, random.uniform(0.3, 0.8)))
        for _ in range(5):
            x, y = random.randint(0, width - 60), random.randint(345, 405)
            fish = self.canvas.create_polygon(x, y, x + 35, y - 15, x + 35, y + 15, fill="#f28f3b", outline="")
            self.relax_items.append((fish, random.uniform(0.8, 1.6)))

    def animate_relax_scene(self):
        stage = self.current_stage()
        if stage is None or stage.kind != "relax":
            return
        width = max(self.canvas.winfo_width(), 900)
        for item, dx in self.relax_items:
            self.canvas.move(item, dx, 0)
            bbox = self.canvas.bbox(item)
            if bbox and bbox[0] > width:
                self.canvas.move(item, -width - 100, 0)
        self.root.after(40, self.animate_relax_scene)

    def show_finished(self):
        text = f"Session finished.\n\nSaved labeled samples: {self.labeled_sample_count}\nRaw: {self.raw_output_path}\nFeatures: {self.features_output_path}"
        ttk.Label(self.content_frame, text=text, wraplength=900, font=("Arial", 16)).pack(anchor="nw")

    def task_detail(self):
        stage = self.current_stage()
        if stage is None:
            return ""
        if stage.kind == "reading":
            return "EEG rhythm reading"
        if stage.kind == "math":
            return self.current_problem
        if stage.kind == "relax":
            return "procedural relaxing scenery"
        return ""

    def drain_samples(self):
        while True:
            try:
                item = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            if item["type"] == "error":
                messagebox.showerror("UDP error", item["message"])
                continue
            self.sample_count += 1
            stage = self.current_stage()
            if stage is None or not stage.collect or stage.label is None:
                continue
            packet = item["row"]
            elapsed = time.time() - self.stage_started_at
            recorded_time = iso_now()
            recorded_unix_time = time.time()
            raw_row = {
                "recorded_time": recorded_time,
                "recorded_unix_time": recorded_unix_time,
                "subject_id": self.subject_id,
                "session_id": self.session_id,
                "stage_name": stage.name,
                "label": stage.label,
                "task_kind": stage.kind,
                "task_detail": self.task_detail(),
                "task_answer": self.current_answer,
                "stage_elapsed_seconds": elapsed,
                "packet_time": packet.get("packet_time"),
                "packet_unix_time": packet.get("packet_unix_time"),
                "packet_size_bytes": packet.get("packet_size_bytes"),
                "decoded_float_count": packet.get("decoded_float_count"),
                "extraction_method": packet.get("extraction_method"),
                "source_host": packet.get("source_host"),
                "source_port": packet.get("source_port"),
            }
            feature_row = {"subject_id": self.subject_id, "session_id": self.session_id, "label": stage.label}
            for band in self.band_names:
                raw_row[band] = packet.get(band)
                feature_row[band] = packet.get(band)
            self.raw_writer.writerow(raw_row)
            self.features_writer.writerow(feature_row)
            self.raw_file.flush()
            self.features_file.flush()
            self.labeled_sample_count += 1

    def tick(self):
        self.drain_samples()
        stage = self.current_stage()
        if stage is not None and self.stage_started_at is not None:
            elapsed = time.time() - self.stage_started_at
            if stage.duration_seconds > 0:
                remaining = max(0, stage.duration_seconds - elapsed)
                self.progress.config(value=min(100, 100 * elapsed / stage.duration_seconds))
                collect_text = "COLLECTING" if stage.collect else "not collecting"
                self.status_label.config(text=f"{collect_text} | label={stage.label or '-'} | remaining={remaining:0.1f}s | UDP port={self.args.recv_port}")
                if elapsed >= stage.duration_seconds:
                    self.next_stage()
            else:
                self.progress.config(value=100)
                self.status_label.config(text="Finished")
        self.sample_label.config(text=f"Samples seen: {self.sample_count} | Labeled saved: {self.labeled_sample_count}")
        self.root.after(100, self.tick)

    def on_close(self):
        self.stop_event.set()
        for handle in (self.raw_file, self.features_file):
            if handle:
                handle.flush()
                handle.close()
        self.raw_file = None
        self.features_file = None
        self.root.destroy()


def run_training_interface(args, band_names):
    root = tk.Tk()
    TrainingInterface(root, args, band_names)
    root.mainloop()


def build_parser():
    parser = argparse.ArgumentParser(description="Collect normalized EEG band values from OpenBCI over UDP")
    parser.add_argument("--interface", action="store_true", help="Launch labeled training GUI instead of plain capture")
    parser.add_argument("--recv-port", type=int, default=12345, help="UDP port to receive OpenBCI binary packets")
    parser.add_argument("--raw-output-csv", required=True, help="Save raw session rows to CSV")
    parser.add_argument("--features-output-csv", help="Save ML-ready features CSV. Default: <subject>-features.csv next to raw output")
    parser.add_argument("--window-output-csv", help="Save overlapped window feature rows to CSV")
    parser.add_argument("--window-seconds", type=float, default=3.0, help="Window size in seconds for overlapped aggregation")
    parser.add_argument("--step-seconds", type=float, default=1.5, help="Step size in seconds for overlap")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Estimated samples per second")
    parser.add_argument("--num-channels", type=int, default=4, help="Deprecated; kept for backward compatibility")
    parser.add_argument("--timeout", type=float, default=10.0, help="UDP receive timeout in seconds")
    parser.add_argument("--max-windows", type=int, default=None, help="Maximum number of samples to capture")
    parser.add_argument("--band-names", default=",".join(DEFAULT_BAND_NAMES), help="Comma-separated names for five values")
    parser.add_argument("--label-schedule", help="CSV schedule file with start_seconds,end_seconds,label")
    parser.add_argument("--default-label", default=None, help="Label outside schedule ranges")
    parser.add_argument("--subject-id", default="subject_001", help="Subject identifier for interface mode")
    parser.add_argument("--session-id", default=None, help="Session id for interface mode. Defaults to timestamp")
    parser.add_argument("--prepare-seconds", type=int, default=10, help="Unlabeled preparation duration")
    parser.add_argument("--focus-seconds", type=int, default=90, help="Focused reading duration")
    parser.add_argument("--math-seconds", type=int, default=90, help="Focused arithmetic duration")
    parser.add_argument("--relax-seconds", type=int, default=90, help="Relaxed scenery duration")
    return parser


def main():
    args = build_parser().parse_args()
    band_names = parse_band_names(args.band_names)
    if args.interface:
        run_training_interface(args, band_names)
        return
    raw_df = receive_openbci_band_power_udp(
        port=args.recv_port,
        num_channels=args.num_channels,
        band_names=band_names,
        timeout=args.timeout,
        max_windows=args.max_windows,
    )
    raw_df.to_csv(args.raw_output_csv, index=False)
    print(f"Saved {len(raw_df)} raw sample rows to {args.raw_output_csv}")
    if args.window_output_csv:
        window_df = make_overlapping_windows(raw_df, args.window_seconds, args.step_seconds, args.sample_rate)
        if args.label_schedule:
            schedule_df = load_label_schedule(args.label_schedule)
            window_df = assign_labels_to_windows(window_df, schedule_df, default_label=args.default_label)
        output_df = window_df.drop(
            columns=[c for c in [
                "window_start_row", "window_end_row", "window_size_samples",
                "window_start_sec", "window_end_sec", "window_midpoint_sec",
            ] if c in window_df.columns],
            errors="ignore",
        )
        output_df.to_csv(args.window_output_csv, index=False)
        print(f"Saved {len(output_df)} overlapped window rows to {args.window_output_csv}")


if __name__ == "__main__":
    main()
