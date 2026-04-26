import argparse
import ast
import csv
import json
import math
import queue
import random
import re
import socket
import struct
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

OPENBCI_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ML_FEATURE_BANDS = ["theta", "alpha", "beta"]
FOCUSED_LABEL = "focused"
RELAXED_LABEL = "relaxed"

READING_SETS = [
    {
        "title": "Focus, EEG, and study data",
        "text": """
Focus, EEG, and study data

A study focus detector is only as good as the data used to train it. EEG cannot read thoughts. It measures electrical activity from the scalp, and that activity is affected by attention, fatigue, eye movement, muscle movement, headset placement, and noise. This means the data collection interface has to control the task state carefully. If the subject is reading, the label should be focused. If the subject is watching a calm scene and not solving anything, the label should be relaxed.

OpenBCI often reports band powers for delta, theta, alpha, beta, and gamma. For this first model, theta, alpha, and beta are the useful model inputs. Delta is more related to sleep and heavy drowsiness. Gamma can be useful in some learning research, but it is also more sensitive to noise and muscle artifacts. The model can still save delta and gamma in the raw file for debugging, but the feature file should stay simple.

The reason for using multiple tasks is that focus is not one exact behavior. Reading uses sustained attention and comprehension. Arithmetic uses working memory and problem solving. Recall questions make the subject treat the reading seriously. A relaxed scene gives contrast. Without a relaxed class, the model only learns what focused examples look like, not what focused is different from.

A good dataset should be balanced and repeatable. The subject should sit still, avoid talking, and keep the headset position consistent. If the same subject is recorded multiple times, each session should be saved separately. Training, testing, and validation data should not all come from the exact same short segment, otherwise the model may look better than it really is.

This interface saves two files. The raw file contains task metadata, packet metadata, and all decoded band powers. It can also include rejected packets for debugging. The feature file contains only the columns needed by the model: subject, session, label, theta, alpha, and beta. If the feature file is empty, that means packets were not received or not decoded into valid band powers.
""".strip(),
        "questions": [
            ("Why should labels be controlled by the interface?", ["Because the task state is known at that moment", "Because labels can be guessed later", "Because beta is always focus", "Because CSV files require labels"], 0),
            ("Which bands are used in the feature file?", ["Theta, alpha, beta", "Delta and gamma", "Only gamma", "All raw bytes"], 0),
            ("Why keep a relaxed class?", ["It gives contrast for the focused class", "It removes noise completely", "It replaces testing", "It makes labels unnecessary"], 0),
            ("What does an empty feature file usually mean?", ["No valid decoded band-power packets were saved", "The model is finished", "The subject was too focused", "The reading was too long"], 0),
        ],
    },
    {
        "title": "Calibration and reliable focus prediction",
        "text": """
Calibration and reliable focus prediction

People do not have identical EEG baselines. One subject may have naturally high alpha when relaxed. Another subject may move more or have different electrode contact. Calibration gives the system a short sample of the current user's relaxed baseline. It does not replace training. It simply helps future predictions be interpreted relative to the subject's own normal state.

The backend can use calibration by averaging relaxed theta, alpha, and beta values. Later, when the frontend sends new band powers, the backend subtracts that baseline before running the classifier. This can reduce user-to-user offsets. It is not magic, and it can make results worse if calibration is collected while the subject is moving, talking, blinking heavily, or already doing a mental task.

A frontend should not need to know how features are calculated. It should send band powers to the backend and receive only a binary classification and confidence. The backend should handle feature extraction, calibration, and the saved model. This keeps the user interface simple and prevents the frontend from depending on machine-learning internals.

Data collection and prediction should use the same assumptions. If the training pipeline uses theta, alpha, beta, and ratios, the backend should use the same transformation. If the labels are focused and relaxed during training, the backend should return focused and relaxed during inference. Consistency matters more than adding complicated models too early.

When a system fails, debugging should start with packets. Are packets arriving at the expected port? Are they UDP or TCP? Are they binary float32 values, JSON, CSV text, or OSC-like data? A collector that silently drops packets is hard to debug. A robust collector should count received packets, decoded packets, rejected packets, and saved labeled samples.
""".strip(),
        "questions": [
            ("What does calibration collect?", ["A relaxed baseline", "A new model", "A gamma-only dataset", "A password"], 0),
            ("What should the frontend receive from prediction?", ["Classification and confidence", "Every feature", "Raw packet bytes", "Training labels"], 0),
            ("Why should training and prediction use the same assumptions?", ["So the model sees the same kind of input", "So files are larger", "So calibration is skipped", "So labels are random"], 0),
            ("What should debugging start with?", ["Packet arrival and decoding", "Changing the model first", "Deleting relaxed data", "Adding gamma"], 0),
        ],
    },
]

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
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in subject_id.strip()) or "subject"


def default_features_path(raw_output_csv, subject_id):
    return Path(raw_output_csv).with_name(f"{safe_subject_filename(subject_id)}-features.csv")


def default_diagnostics_path(raw_output_csv, subject_id):
    return Path(raw_output_csv).with_name(f"{safe_subject_filename(subject_id)}-diagnostics.csv")


def normalize_band_dict(values, source="unknown"):
    lower = {str(k).strip().lower(): v for k, v in values.items()}
    if all(band in lower for band in OPENBCI_BANDS):
        bands = {band: float(lower[band]) for band in OPENBCI_BANDS}
    elif all(band in lower for band in ML_FEATURE_BANDS):
        bands = {band: None for band in OPENBCI_BANDS}
        for band in ML_FEATURE_BANDS:
            bands[band] = float(lower[band])
    else:
        return None
    if not all(bands[band] is None or math.isfinite(float(bands[band])) for band in OPENBCI_BANDS):
        return None
    row = {
        "packet_time": iso_now(),
        "packet_unix_time": time.time(),
        "packet_size_bytes": None,
        "decoded_float_count": None,
        "extraction_method": source,
        "decode_status": "decoded",
        "decode_error": "",
        "raw_preview": "",
    }
    row.update(bands)
    return row


def choose_band_values(floats, tolerance=0.08):
    if len(floats) < 3:
        return None, "too_few_floats"

    # Prefer a normalized OpenBCI-style 5-band window: delta, theta, alpha, beta, gamma.
    for start in range(len(floats) - 5, -1, -1):
        window = [float(v) for v in floats[start:start + 5]]
        if all(math.isfinite(v) and 0 <= v <= 1 for v in window) and abs(sum(window) - 1.0) <= tolerance:
            return dict(zip(OPENBCI_BANDS, window)), f"normalized_5_band_window_start_{start}"

    # Accept any last sane 5-band sequence even if the sum is not exactly 1.
    sane = [float(v) for v in floats if math.isfinite(v) and 0 <= v <= 1]
    if len(sane) >= 5:
        return dict(zip(OPENBCI_BANDS, sane[-5:])), "last_5_sane_values"

    # Accept theta/alpha/beta only.
    if len(sane) >= 3:
        return {"delta": None, "theta": sane[-3], "alpha": sane[-2], "beta": sane[-1], "gamma": None}, "last_3_sane_values_theta_alpha_beta"

    return None, "no_valid_band_values"


def unpack_float32_packets(data):
    usable_size = len(data) - (len(data) % 4)
    if usable_size <= 0:
        return []
    out = []
    for endian, name in ((">", "float32_be"), ("<", "float32_le")):
        try:
            values = [float(v) for v in struct.unpack(f"{endian}{usable_size // 4}f", data[:usable_size])]
            out.append((name, values))
        except struct.error:
            pass
    return out


def extract_numbers_from_text(text):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]


def extract_from_json_text(text):
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        if "band_powers" in payload:
            source = payload["band_powers"]
            if isinstance(source, dict):
                return normalize_band_dict(source, "json_band_powers_object")
            if isinstance(source, list):
                bands, method = choose_band_values([float(v) for v in source])
                if bands:
                    return normalize_band_dict(bands, f"json_band_powers_list_{method}")
        return normalize_band_dict(payload, "json_band_object")
    if isinstance(payload, list):
        bands, method = choose_band_values([float(v) for v in payload])
        if bands:
            return normalize_band_dict(bands, f"json_list_{method}")
    return None


def extract_openbci_bands(data):
    raw_preview = data[:160].hex()

    # 1) Try UTF-8 text formats: JSON, CSV, Python-list-ish, OSC-address plus numbers.
    try:
        text = data.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError:
        text = ""

    if text:
        row = extract_from_json_text(text)
        if row:
            row["packet_size_bytes"] = len(data)
            row["raw_preview"] = text[:160]
            return row

        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                bands, method = choose_band_values([float(v) for v in parsed])
                if bands:
                    row = normalize_band_dict(bands, f"literal_{method}")
                    row["packet_size_bytes"] = len(data)
                    row["raw_preview"] = text[:160]
                    return row
            elif isinstance(parsed, dict):
                row = normalize_band_dict(parsed, "literal_dict")
                if row:
                    row["packet_size_bytes"] = len(data)
                    row["raw_preview"] = text[:160]
                    return row
        except Exception:
            pass

        numbers = extract_numbers_from_text(text)
        bands, method = choose_band_values(numbers)
        if bands:
            row = normalize_band_dict(bands, f"text_numbers_{method}")
            row["packet_size_bytes"] = len(data)
            row["decoded_float_count"] = len(numbers)
            row["raw_preview"] = text[:160]
            return row

    # 2) Try binary float32, both endian directions.
    for float_method, floats in unpack_float32_packets(data):
        bands, method = choose_band_values(floats)
        if bands:
            row = normalize_band_dict(bands, f"{float_method}_{method}")
            row["packet_size_bytes"] = len(data)
            row["decoded_float_count"] = len(floats)
            row["raw_preview"] = raw_preview
            return row

    return {
        "packet_time": iso_now(),
        "packet_unix_time": time.time(),
        "packet_size_bytes": len(data),
        "decoded_float_count": 0,
        "extraction_method": "none",
        "decode_status": "rejected",
        "decode_error": "could_not_extract_theta_alpha_beta_or_5_band_values",
        "raw_preview": raw_preview,
        **{band: None for band in OPENBCI_BANDS},
    }


class UDPReader(threading.Thread):
    def __init__(self, port, output_queue, stop_event, bind_host="0.0.0.0"):
        super().__init__(daemon=True)
        self.port = port
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.bind_host = bind_host

    def run(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind((self.bind_host, self.port))
                sock.settimeout(0.5)
                self.output_queue.put({"type": "status", "message": f"UDP listening on {self.bind_host}:{self.port}"})
                while not self.stop_event.is_set():
                    try:
                        data, address = sock.recvfrom(65535)
                    except socket.timeout:
                        continue
                    except OSError as exc:
                        self.output_queue.put({"type": "error", "message": str(exc)})
                        break
                    row = extract_openbci_bands(data)
                    row["source_host"] = address[0]
                    row["source_port"] = address[1]
                    self.output_queue.put({"type": "packet", "row": row})
        except Exception as exc:
            self.output_queue.put({"type": "error", "message": f"UDP reader failed: {exc}"})


class TrainingInterface:
    def __init__(self, root, args, feature_bands):
        self.root = root
        self.args = args
        self.feature_bands = feature_bands
        self.subject_id = args.subject_id
        self.session_id = args.session_id or time.strftime("%Y%m%d_%H%M%S")
        self.raw_output_path = Path(args.raw_output_csv)
        self.features_output_path = Path(args.features_output_csv or default_features_path(args.raw_output_csv, self.subject_id))
        self.diagnostics_output_path = Path(args.diagnostics_csv or default_diagnostics_path(args.raw_output_csv, self.subject_id))
        self.reading_set = random.choice(READING_SETS) if args.reading_index is None else READING_SETS[args.reading_index]
        self.reading_questions = self.reading_set["questions"][:]
        random.shuffle(self.reading_questions)
        self.sample_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.packet_count = 0
        self.decoded_packet_count = 0
        self.rejected_packet_count = 0
        self.labeled_sample_count = 0
        self.raw_saved_count = 0
        self.stage_index = -1
        self.stage_started_at = None
        self.current_problem = ""
        self.current_answer = ""
        self.quiz_index = 0
        self.quiz_selected_text = ""
        self.quiz_correct_text = ""
        self.quiz_is_correct = ""
        self.relax_items = []
        self.last_status = "Starting"
        self.stages = [
            Stage("Prepare", None, "instruction", args.prepare_seconds, False),
            Stage("Focused reading", FOCUSED_LABEL, "reading", args.focus_seconds, True),
            Stage("Reading recall", FOCUSED_LABEL, "quiz", args.quiz_seconds, True),
            Stage("Focused arithmetic", FOCUSED_LABEL, "math", args.math_seconds, True),
            Stage("Relaxed scenery", RELAXED_LABEL, "relax", args.relax_seconds, True),
            Stage("Finished", None, "finished", 0, False),
        ]
        self.root.title("NeuroFocus EEG Training Collector - Robust")
        self.root.geometry("1080x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.open_csvs()
        self.reader = UDPReader(args.recv_port, self.sample_queue, self.stop_event, bind_host=args.bind_host)
        self.reader.start()
        self.next_stage()
        self.root.after(100, self.tick)

    def build_ui(self):
        self.top = ttk.Frame(self.root, padding=12)
        self.top.pack(fill="x")
        self.title_label = ttk.Label(self.top, text="", font=("Arial", 22, "bold"))
        self.title_label.pack(anchor="w")
        self.status_label = ttk.Label(self.top, text="")
        self.status_label.pack(anchor="w")
        self.progress = ttk.Progressbar(self.top, maximum=100)
        self.progress.pack(fill="x", pady=8)
        self.content = ttk.Frame(self.root, padding=12)
        self.content.pack(fill="both", expand=True)
        bottom = ttk.Frame(self.root, padding=12)
        bottom.pack(fill="x")
        self.sample_label = ttk.Label(bottom, text="")
        self.sample_label.pack(side="left")
        ttk.Button(bottom, text="Next stage", command=self.next_stage).pack(side="right", padx=8)
        ttk.Button(bottom, text="Stop", command=self.on_close).pack(side="right")

    def open_writer(self, path, fieldnames):
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        handle = path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
            handle.flush()
        return handle, writer

    def open_csvs(self):
        raw_fields = [
            "recorded_time", "recorded_unix_time", "subject_id", "session_id", "stage_name", "label",
            "task_kind", "task_detail", "task_answer", "question", "selected_answer", "correct_answer", "is_correct",
            "reading_title", "stage_elapsed_seconds", "decode_status", "decode_error", "packet_time", "packet_unix_time",
            "packet_size_bytes", "decoded_float_count", "extraction_method", "source_host", "source_port", "raw_preview", *OPENBCI_BANDS,
        ]
        feature_fields = ["subject_id", "session_id", "label", *self.feature_bands]
        diagnostic_fields = ["time", "event", "message", "packets", "decoded", "rejected", "raw_saved", "features_saved"]
        self.raw_file, self.raw_writer = self.open_writer(self.raw_output_path, raw_fields)
        self.features_file, self.features_writer = self.open_writer(self.features_output_path, feature_fields)
        self.diag_file, self.diag_writer = self.open_writer(self.diagnostics_output_path, diagnostic_fields)
        self.write_diagnostic("startup", f"raw={self.raw_output_path}; features={self.features_output_path}; diagnostics={self.diagnostics_output_path}")

    def write_diagnostic(self, event, message):
        self.diag_writer.writerow({
            "time": iso_now(),
            "event": event,
            "message": message,
            "packets": self.packet_count,
            "decoded": self.decoded_packet_count,
            "rejected": self.rejected_packet_count,
            "raw_saved": self.raw_saved_count,
            "features_saved": self.labeled_sample_count,
        })
        self.diag_file.flush()

    def clear_content(self):
        for widget in self.content.winfo_children():
            widget.destroy()

    def current_stage(self):
        return self.stages[self.stage_index] if 0 <= self.stage_index < len(self.stages) else None

    def next_stage(self):
        self.stage_index += 1
        if self.stage_index >= len(self.stages):
            self.on_close()
            return
        stage = self.current_stage()
        self.stage_started_at = time.time()
        self.current_answer = ""
        self.write_diagnostic("stage", f"entered {stage.name}; collect={stage.collect}; label={stage.label}")
        self.clear_content()
        self.title_label.config(text=stage.name)
        getattr(self, f"show_{stage.kind}")()

    def show_instruction(self):
        text = (
            f"Listening for UDP packets on {self.args.bind_host}:{self.args.recv_port}\n"
            f"Raw file: {self.raw_output_path}\n"
            f"Features file: {self.features_output_path}\n"
            f"Diagnostics file: {self.diagnostics_output_path}\n\n"
            "If packets arrive but features stay at 0, the packet format is not being decoded. "
            "Check the diagnostics file and raw_preview column.\n\n"
            f"Selected reading: {self.reading_set['title']}"
        )
        ttk.Label(self.content, text=text, wraplength=980, font=("Arial", 15)).pack(anchor="nw")

    def show_reading(self):
        ttk.Label(self.content, text=f"{self.reading_set['title']} — keep reading until the timer ends.", font=("Arial", 15, "bold"), wraplength=980).pack(anchor="nw", pady=(0, 10))
        box = tk.Text(self.content, wrap="word", font=("Arial", 14), height=26)
        box.insert("1.0", self.reading_set["text"])
        box.config(state="disabled")
        box.pack(fill="both", expand=True)

    def show_quiz(self):
        self.show_quiz_question()

    def show_quiz_question(self):
        self.clear_content()
        question, choices, answer = self.reading_questions[self.quiz_index % len(self.reading_questions)]
        self.quiz_correct_text = choices[answer]
        self.quiz_selected_text = ""
        self.quiz_is_correct = ""
        ttk.Label(self.content, text=question, wraplength=980, font=("Arial", 18, "bold")).pack(anchor="nw", pady=12)
        for idx, choice in enumerate(choices):
            ttk.Button(self.content, text=choice, command=lambda i=idx: self.answer_quiz(i)).pack(fill="x", pady=5)
        self.quiz_feedback = ttk.Label(self.content, text="", font=("Arial", 14))
        self.quiz_feedback.pack(anchor="nw", pady=12)

    def answer_quiz(self, choice_index):
        question, choices, answer = self.reading_questions[self.quiz_index % len(self.reading_questions)]
        self.quiz_selected_text = choices[choice_index]
        self.quiz_correct_text = choices[answer]
        self.quiz_is_correct = str(choice_index == answer)
        self.current_answer = self.quiz_selected_text
        self.quiz_feedback.config(text="Correct" if choice_index == answer else f"Correct answer: {self.quiz_correct_text}")
        self.quiz_index += 1
        self.root.after(900, self.show_quiz_question)

    def make_problem(self):
        a, b, c = random.randint(12, 29), random.randint(3, 12), random.randint(10, 60)
        if random.choice([True, False]):
            self.current_problem = f"{a} x {b}"
            self.problem_answer = a * b
        else:
            self.current_problem = f"({a} x {b}) - {c}"
            self.problem_answer = a * b - c

    def show_math(self):
        self.make_problem()
        ttk.Label(self.content, text="Solve arithmetic. Press Enter.", font=("Arial", 15, "bold")).pack(anchor="nw")
        self.math_problem_label = ttk.Label(self.content, text=self.current_problem, font=("Arial", 34, "bold"))
        self.math_problem_label.pack(pady=20)
        self.answer_var = tk.StringVar()
        entry = ttk.Entry(self.content, textvariable=self.answer_var, font=("Arial", 24), justify="center")
        entry.pack(ipady=8)
        entry.bind("<Return>", self.submit_math_answer)
        entry.focus_set()
        self.math_feedback_label = ttk.Label(self.content, text="", font=("Arial", 14))
        self.math_feedback_label.pack(pady=12)

    def submit_math_answer(self, event=None):
        text = self.answer_var.get().strip()
        self.current_answer = text
        self.math_feedback_label.config(text="Correct" if text.lstrip("-").isdigit() and int(text) == self.problem_answer else f"Correct answer was {self.problem_answer}")
        self.answer_var.set("")
        self.make_problem()
        self.math_problem_label.config(text=self.current_problem)

    def show_relax(self):
        ttk.Label(self.content, text="Relax and watch the forest scenery.", font=("Arial", 15, "bold")).pack(anchor="nw")
        self.canvas = tk.Canvas(self.content, width=980, height=500, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.draw_forest_scene()
        self.animate_relax_scene()

    def draw_forest_scene(self):
        self.canvas.delete("all")
        width, height = 980, 500
        self.canvas.create_rectangle(0, 0, width, height, fill="#b9defb", outline="")
        self.canvas.create_oval(760, 45, 845, 130, fill="#fff1a8", outline="")
        self.canvas.create_rectangle(0, 300, width, height, fill="#6fb36b", outline="")
        self.canvas.create_rectangle(0, 390, width, height, fill="#5ca6cf", outline="")
        for x in range(-40, width + 80, 55):
            self.canvas.create_polygon(x, 300, x + 35, 175, x + 70, 300, fill="#477f55", outline="")
        for x in range(-20, width + 80, 70):
            self.canvas.create_polygon(x, 330, x + 45, 190, x + 90, 330, fill="#2f6b46", outline="")
        self.relax_items = []
        for _ in range(7):
            x, y = random.randint(0, width - 110), random.randint(35, 140)
            cloud_parts = [
                self.canvas.create_oval(x, y, x + 70, y + 35, fill="white", outline=""),
                self.canvas.create_oval(x + 35, y - 12, x + 105, y + 35, fill="white", outline=""),
            ]
            self.relax_items.append({"items": cloud_parts, "dx": random.uniform(0.25, 0.7), "wrap": width + 140})
        for _ in range(8):
            x, y = random.randint(0, width - 40), random.randint(410, 480)
            ripple = self.canvas.create_arc(x, y, x + 45, y + 20, start=0, extent=180, style="arc", outline="#d8f6ff", width=1)
            self.relax_items.append({"items": [ripple], "dx": random.uniform(0.4, 1.0), "wrap": width + 60})

    def animate_relax_scene(self):
        stage = self.current_stage()
        if stage is None or stage.kind != "relax" or not hasattr(self, "canvas"):
            return
        width = max(self.canvas.winfo_width(), 980)
        for obj in self.relax_items:
            for item in obj["items"]:
                self.canvas.move(item, obj.get("dx", 0.5), obj.get("dy", 0.0))
            bbox = self.canvas.bbox(obj["items"][0])
            if bbox and bbox[0] > obj.get("wrap", width + 100):
                for item in obj["items"]:
                    self.canvas.move(item, -width - 140, 0)
        self.root.after(40, self.animate_relax_scene)

    def show_finished(self):
        ttk.Label(self.content, text=f"Session finished.\nPackets: {self.packet_count}\nDecoded: {self.decoded_packet_count}\nRejected: {self.rejected_packet_count}\nRaw rows: {self.raw_saved_count}\nFeature rows: {self.labeled_sample_count}\nRaw: {self.raw_output_path}\nFeatures: {self.features_output_path}\nDiagnostics: {self.diagnostics_output_path}", wraplength=900, font=("Arial", 16)).pack(anchor="nw")

    def task_detail(self):
        stage = self.current_stage()
        if stage.kind == "reading":
            return self.reading_set["title"]
        if stage.kind == "quiz":
            return self.reading_questions[self.quiz_index % len(self.reading_questions)][0]
        if stage.kind == "math":
            return self.current_problem
        if stage.kind == "relax":
            return "animated forest relaxing scenery"
        return ""

    def build_raw_row(self, packet, stage):
        return {
            "recorded_time": iso_now(),
            "recorded_unix_time": time.time(),
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "stage_name": stage.name if stage else "",
            "label": stage.label if stage and stage.label else "",
            "task_kind": stage.kind if stage else "",
            "task_detail": self.task_detail() if stage else "",
            "task_answer": self.current_answer,
            "question": self.task_detail() if stage and stage.kind == "quiz" else "",
            "selected_answer": self.quiz_selected_text if stage and stage.kind == "quiz" else "",
            "correct_answer": self.quiz_correct_text if stage and stage.kind == "quiz" else "",
            "is_correct": self.quiz_is_correct if stage and stage.kind == "quiz" else "",
            "reading_title": self.reading_set["title"] if stage and stage.kind in {"reading", "quiz"} else "",
            "stage_elapsed_seconds": time.time() - self.stage_started_at if self.stage_started_at else "",
            **packet,
        }

    def drain_samples(self):
        while True:
            try:
                item = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            if item["type"] == "status":
                self.last_status = item["message"]
                self.write_diagnostic("status", item["message"])
                continue
            if item["type"] == "error":
                self.last_status = item["message"]
                self.write_diagnostic("error", item["message"])
                messagebox.showerror("UDP error", item["message"])
                continue
            packet = item["row"]
            self.packet_count += 1
            if packet.get("decode_status") == "decoded":
                self.decoded_packet_count += 1
            else:
                self.rejected_packet_count += 1
            stage = self.current_stage()
            if stage is None or not stage.collect:
                continue

            if packet.get("decode_status") == "decoded" or self.args.save_rejected_raw:
                self.raw_writer.writerow(self.build_raw_row(packet, stage))
                self.raw_file.flush()
                self.raw_saved_count += 1

            if packet.get("decode_status") != "decoded" or stage.label is None:
                continue
            if not all(packet.get(band) is not None for band in self.feature_bands):
                continue
            feature_row = {"subject_id": self.subject_id, "session_id": self.session_id, "label": stage.label}
            for band in self.feature_bands:
                feature_row[band] = packet.get(band)
            self.features_writer.writerow(feature_row)
            self.features_file.flush()
            self.labeled_sample_count += 1

    def tick(self):
        self.drain_samples()
        stage = self.current_stage()
        if stage and self.stage_started_at:
            elapsed = time.time() - self.stage_started_at
            if stage.duration_seconds > 0:
                remaining = max(0, stage.duration_seconds - elapsed)
                self.progress.config(value=min(100, 100 * elapsed / stage.duration_seconds))
                self.status_label.config(text=f"{'COLLECTING' if stage.collect else 'not collecting'} | label={stage.label or '-'} | remaining={remaining:0.1f}s | {self.last_status}")
                if elapsed >= stage.duration_seconds:
                    self.next_stage()
        self.sample_label.config(text=f"Packets: {self.packet_count} | Decoded: {self.decoded_packet_count} | Rejected: {self.rejected_packet_count} | Raw saved: {self.raw_saved_count} | Features saved: {self.labeled_sample_count}")
        self.root.after(100, self.tick)

    def on_close(self):
        self.write_diagnostic("shutdown", "closing training interface")
        self.stop_event.set()
        for handle in (getattr(self, "raw_file", None), getattr(self, "features_file", None), getattr(self, "diag_file", None)):
            if handle:
                handle.flush()
                handle.close()
        self.root.destroy()


def parse_band_names(text):
    names = [name.strip() for name in text.split(",") if name.strip()] if text else ML_FEATURE_BANDS
    invalid = [name for name in names if name not in OPENBCI_BANDS]
    if invalid:
        raise ValueError(f"Unknown band names: {invalid}")
    return names


def build_parser():
    parser = argparse.ArgumentParser(description="Robust labeled EEG training interface with packet diagnostics")
    parser.add_argument("--recv-port", type=int, default=12345)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--raw-output-csv", required=True)
    parser.add_argument("--features-output-csv")
    parser.add_argument("--diagnostics-csv")
    parser.add_argument("--feature-bands", default=",".join(ML_FEATURE_BANDS))
    parser.add_argument("--subject-id", default="subject_001")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--reading-index", type=int, default=None, help="Select a fixed reading set index. Defaults to random.")
    parser.add_argument("--prepare-seconds", type=int, default=10)
    parser.add_argument("--focus-seconds", type=int, default=180)
    parser.add_argument("--quiz-seconds", type=int, default=60)
    parser.add_argument("--math-seconds", type=int, default=90)
    parser.add_argument("--relax-seconds", type=int, default=90)
    parser.add_argument("--save-rejected-raw", action="store_true", default=True, help="Save rejected packets into the raw CSV for debugging. Default: true.")
    return parser


def main():
    args = build_parser().parse_args()
    feature_bands = parse_band_names(args.feature_bands)
    root = tk.Tk()
    TrainingInterface(root, args, feature_bands)
    root.mainloop()


if __name__ == "__main__":
    main()
