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

OPENBCI_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
ML_FEATURE_BANDS = ["theta", "alpha", "beta"]
FOCUSED_LABEL = "focused"
RELAXED_LABEL = "relaxed"

READING_TEXT = """
How focus changes during studying

Studying is not one simple state. A student may read, remember, calculate, compare ideas, and check answers during the same session. From outside all of this may look like focus, but the brain is doing different work from moment to moment. EEG is useful here because it gives a rough real-time signal from the scalp. It cannot read thoughts and it should not be treated as a perfect attention detector. It is only useful when the data is collected carefully and labeled honestly.

OpenBCI groups activity into common frequency bands. Delta is the slowest and is usually more useful for sleep or heavy drowsiness than normal studying. Theta is also slow. It can increase when a person is tired, but it can also appear during memory work and internal concentration. Alpha is often stronger during relaxed wakefulness, especially when the eyes are closed or the subject is calm. During demanding visual or mental work, alpha often becomes lower compared with rest. Beta is faster and is commonly linked with alertness, active thinking, and problem solving. Gamma is faster again, but for this first focus detector it is less practical because it can be noisy and can mix with muscle activity.

This is why the model should not depend on beta alone. Beta can rise during thinking, but also during tension or movement. Alpha can drop during attention, but lighting and eye behavior can also change it. Theta can be meaningful during memory work, but it can also rise when the subject gets sleepy. Ratios help because they compare bands with each other. Beta divided by alpha can show active thinking relative to relaxed wakefulness. Beta divided by theta can help separate alert effort from drifting or sleepy states. Theta divided by alpha adds context about relaxation and internal focus.

The most important part of this project is the labels. If a row is labeled focused, the subject should actually be doing a known focus task at that moment. If a row is labeled relaxed, the subject should be in a low-demand state. Guessing labels afterward is risky because the subject may change state quickly. The interface should collect only when the state is known. Reading gives a study-like focus state. Arithmetic gives a problem-solving focus state. A calm scenery stage gives a relaxed comparison state.

The reading task must be long enough that the subject cannot finish early. If the text is too short, the final part of the stage may be labeled focused reading even though the subject is just waiting. That would make the training data worse. A longer text keeps the subject engaged for the full timer. Questions after the reading are also useful because they make the subject treat the text seriously. The goal is not to grade the subject. The goal is to make the focus label more believable.

Arithmetic gives another useful focus state. Multiplication and multi-step calculations force the subject to hold numbers in working memory and update intermediate results. This is close to what students do when solving exercises. The answer correctness is useful metadata, but it is not the same thing as the EEG label. A wrong answer can still happen during real focus.

The relaxed condition should be low effort. A calm scene with slow movement is enough for early testing. Later the project can use open-license nature or animal videos, but the task should stay simple. It should not ask the subject to read, calculate, or memorize details.

A good first model should be personal and simple. EEG differs between people, electrodes, hair, movement, fatigue, and hardware settings. A model trained on one subject may not work for another. The first goal is not a universal focus detector. The first goal is to collect clean labeled data, train a small model, and check whether it performs better than chance on held-out samples.
""".strip()

READING_QUESTIONS = [
    ("Why should labels come from the current task stage?", ["Because EEG state changes quickly", "Because labels are optional", "Because beta is always focus", "Because OpenBCI has no ports"], 0),
    ("Which bands are used by the first focus model?", ["Delta and gamma", "Theta, alpha, and beta", "Only beta", "All bands only"], 1),
    ("Why should the reading text be long?", ["To avoid finishing early while still recording focused data", "To remove the relaxed class", "To increase gamma", "To make CSV smaller"], 0),
    ("Why are beta/alpha and beta/theta useful?", ["They compare active thinking against context bands", "They remove all noise", "They replace labels", "They decode thoughts"], 0),
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


def unpack_float32_be_packet(data):
    usable_size = len(data) - (len(data) % 4)
    if usable_size <= 0:
        return []
    return [float(v) for v in struct.unpack(f">{usable_size // 4}f", data[:usable_size])]


def choose_band_values(floats, tolerance=0.08):
    if len(floats) < 5:
        return None, "too_few_floats"
    for start in range(len(floats) - 5, -1, -1):
        window = [float(v) for v in floats[start:start + 5]]
        if all(math.isfinite(v) and 0 <= v <= 1 for v in window) and abs(sum(window) - 1.0) <= tolerance:
            return window, f"normalized_window_start_{start}"
    sane = [float(v) for v in floats if math.isfinite(v) and 0 <= v <= 1]
    if len(sane) >= 5 and sum(sane[-5:]) > 0:
        total = sum(sane[-5:])
        return [v / total for v in sane[-5:]], "normalized_last_sane_values"
    return None, "no_valid_normalized_window"


def extract_openbci_bands(data):
    floats = unpack_float32_be_packet(data)
    values, method = choose_band_values(floats)
    if values is None:
        return None
    row = {
        "packet_time": iso_now(),
        "packet_unix_time": time.time(),
        "packet_size_bytes": len(data),
        "decoded_float_count": len(floats),
        "extraction_method": method,
    }
    for band, value in zip(OPENBCI_BANDS, values):
        row[band] = value
    return row


class UDPReader(threading.Thread):
    def __init__(self, port, output_queue, stop_event):
        super().__init__(daemon=True)
        self.port = port
        self.output_queue = output_queue
        self.stop_event = stop_event

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
                row = extract_openbci_bands(data)
                if row:
                    row["source_host"] = address[0]
                    row["source_port"] = address[1]
                    self.output_queue.put({"type": "sample", "row": row})


class TrainingInterface:
    def __init__(self, root, args, feature_bands):
        self.root = root
        self.args = args
        self.feature_bands = feature_bands
        self.subject_id = args.subject_id
        self.session_id = args.session_id or time.strftime("%Y%m%d_%H%M%S")
        self.raw_output_path = Path(args.raw_output_csv)
        self.features_output_path = Path(args.features_output_csv or default_features_path(args.raw_output_csv, self.subject_id))
        self.sample_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.sample_count = 0
        self.labeled_sample_count = 0
        self.stage_index = -1
        self.stage_started_at = None
        self.current_problem = ""
        self.current_answer = ""
        self.quiz_index = 0
        self.quiz_selected_text = ""
        self.quiz_correct_text = ""
        self.quiz_is_correct = ""
        self.stages = [
            Stage("Prepare", None, "instruction", args.prepare_seconds, False),
            Stage("Focused reading", FOCUSED_LABEL, "reading", args.focus_seconds, True),
            Stage("Reading recall", FOCUSED_LABEL, "quiz", args.quiz_seconds, True),
            Stage("Focused arithmetic", FOCUSED_LABEL, "math", args.math_seconds, True),
            Stage("Relaxed scenery", RELAXED_LABEL, "relax", args.relax_seconds, True),
            Stage("Finished", None, "finished", 0, False),
        ]
        self.root.title("NeuroFocus EEG Training Collector")
        self.root.geometry("1000x760")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.open_csvs()
        self.reader = UDPReader(args.recv_port, self.sample_queue, self.stop_event)
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
        self.sample_label = ttk.Label(bottom, text="Samples seen: 0 | Labeled saved: 0")
        self.sample_label.pack(side="left")
        ttk.Button(bottom, text="Next stage", command=self.next_stage).pack(side="right", padx=8)
        ttk.Button(bottom, text="Stop", command=self.on_close).pack(side="right")

    def open_writer(self, path, fieldnames):
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        handle = path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
            handle.flush()
        return handle, writer

    def open_csvs(self):
        raw_fields = [
            "recorded_time", "recorded_unix_time", "subject_id", "session_id", "stage_name", "label",
            "task_kind", "task_detail", "task_answer", "question", "selected_answer", "correct_answer", "is_correct",
            "stage_elapsed_seconds", "packet_time", "packet_unix_time", "packet_size_bytes", "decoded_float_count",
            "extraction_method", "source_host", "source_port", *OPENBCI_BANDS,
        ]
        feature_fields = ["subject_id", "session_id", "label", *self.feature_bands]
        self.raw_file, self.raw_writer = self.open_writer(self.raw_output_path, raw_fields)
        self.features_file, self.features_writer = self.open_writer(self.features_output_path, feature_fields)

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
        self.clear_content()
        self.title_label.config(text=stage.name)
        getattr(self, f"show_{stage.kind}")()

    def show_instruction(self):
        text = f"OpenBCI UDP port: {self.args.recv_port}\nRaw: {self.raw_output_path}\nFeatures: {self.features_output_path}\nFeature bands: {', '.join(self.feature_bands)}\nLabels: {FOCUSED_LABEL}, {RELAXED_LABEL}\n\nThe reading is intentionally long. Keep reading until time ends; questions follow."
        ttk.Label(self.content, text=text, wraplength=920, font=("Arial", 16)).pack(anchor="nw")

    def show_reading(self):
        ttk.Label(self.content, text="Keep reading until the timer ends. Recall questions come next.", font=("Arial", 15, "bold")).pack(anchor="nw", pady=(0, 10))
        box = tk.Text(self.content, wrap="word", font=("Arial", 14), height=24)
        box.insert("1.0", READING_TEXT)
        box.config(state="disabled")
        box.pack(fill="both", expand=True)

    def show_quiz(self):
        self.show_quiz_question()

    def show_quiz_question(self):
        self.clear_content()
        question, choices, answer = READING_QUESTIONS[self.quiz_index % len(READING_QUESTIONS)]
        self.quiz_correct_text = choices[answer]
        self.quiz_selected_text = ""
        self.quiz_is_correct = ""
        ttk.Label(self.content, text=question, wraplength=900, font=("Arial", 18, "bold")).pack(anchor="nw", pady=12)
        for idx, choice in enumerate(choices):
            ttk.Button(self.content, text=choice, command=lambda i=idx: self.answer_quiz(i)).pack(fill="x", pady=5)
        self.quiz_feedback = ttk.Label(self.content, text="", font=("Arial", 14))
        self.quiz_feedback.pack(anchor="nw", pady=12)

    def answer_quiz(self, choice_index):
        question, choices, answer = READING_QUESTIONS[self.quiz_index % len(READING_QUESTIONS)]
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
        ttk.Label(self.content, text="Relax and watch the scenery.", font=("Arial", 15, "bold")).pack(anchor="nw")
        canvas = tk.Canvas(self.content, width=900, height=430, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_rectangle(0, 0, 900, 430, fill="#b7ddff", outline="")
        canvas.create_oval(650, 45, 730, 125, fill="#fff2a8", outline="")
        canvas.create_rectangle(0, 280, 900, 430, fill="#8ed081", outline="")

    def show_finished(self):
        ttk.Label(self.content, text=f"Session finished.\nSaved labeled samples: {self.labeled_sample_count}\nRaw: {self.raw_output_path}\nFeatures: {self.features_output_path}", wraplength=900, font=("Arial", 16)).pack(anchor="nw")

    def task_detail(self):
        stage = self.current_stage()
        if stage.kind == "reading":
            return "long EEG/focus reading"
        if stage.kind == "quiz":
            return READING_QUESTIONS[self.quiz_index % len(READING_QUESTIONS)][0]
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
            raw_row = {
                "recorded_time": iso_now(),
                "recorded_unix_time": time.time(),
                "subject_id": self.subject_id,
                "session_id": self.session_id,
                "stage_name": stage.name,
                "label": stage.label,
                "task_kind": stage.kind,
                "task_detail": self.task_detail(),
                "task_answer": self.current_answer,
                "question": self.task_detail() if stage.kind == "quiz" else "",
                "selected_answer": self.quiz_selected_text if stage.kind == "quiz" else "",
                "correct_answer": self.quiz_correct_text if stage.kind == "quiz" else "",
                "is_correct": self.quiz_is_correct if stage.kind == "quiz" else "",
                "stage_elapsed_seconds": time.time() - self.stage_started_at,
                "packet_time": packet.get("packet_time"),
                "packet_unix_time": packet.get("packet_unix_time"),
                "packet_size_bytes": packet.get("packet_size_bytes"),
                "decoded_float_count": packet.get("decoded_float_count"),
                "extraction_method": packet.get("extraction_method"),
                "source_host": packet.get("source_host"),
                "source_port": packet.get("source_port"),
            }
            for band in OPENBCI_BANDS:
                raw_row[band] = packet.get(band)
            feature_row = {"subject_id": self.subject_id, "session_id": self.session_id, "label": stage.label}
            for band in self.feature_bands:
                feature_row[band] = packet.get(band)
            self.raw_writer.writerow(raw_row)
            self.features_writer.writerow(feature_row)
            self.raw_file.flush()
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
                self.status_label.config(text=f"{'COLLECTING' if stage.collect else 'not collecting'} | label={stage.label or '-'} | remaining={remaining:0.1f}s")
                if elapsed >= stage.duration_seconds:
                    self.next_stage()
        self.sample_label.config(text=f"Samples seen: {self.sample_count} | Labeled saved: {self.labeled_sample_count}")
        self.root.after(100, self.tick)

    def on_close(self):
        self.stop_event.set()
        for handle in (getattr(self, "raw_file", None), getattr(self, "features_file", None)):
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
    parser = argparse.ArgumentParser(description="Improved labeled EEG training interface")
    parser.add_argument("--recv-port", type=int, default=12345)
    parser.add_argument("--raw-output-csv", required=True)
    parser.add_argument("--features-output-csv")
    parser.add_argument("--feature-bands", default=",".join(ML_FEATURE_BANDS))
    parser.add_argument("--subject-id", default="subject_001")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--prepare-seconds", type=int, default=10)
    parser.add_argument("--focus-seconds", type=int, default=180)
    parser.add_argument("--quiz-seconds", type=int, default=60)
    parser.add_argument("--math-seconds", type=int, default=90)
    parser.add_argument("--relax-seconds", type=int, default=90)
    return parser


def main():
    args = build_parser().parse_args()
    feature_bands = parse_band_names(args.feature_bands)
    root = tk.Tk()
    TrainingInterface(root, args, feature_bands)
    root.mainloop()


if __name__ == "__main__":
    main()
