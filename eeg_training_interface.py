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

READING_SETS = [
    {
        "title": "How focus changes during studying",
        "text": """
How focus changes during studying

Studying is not one simple state. A student may read, remember, calculate, compare ideas, and check answers during the same session. From outside all of this may look like focus, but the brain is doing different work from moment to moment. EEG is useful here because it gives a rough real-time signal from the scalp. It cannot read thoughts and it should not be treated as a perfect attention detector. It is only useful when the data is collected carefully and labeled honestly.

OpenBCI groups activity into common frequency bands. Delta is the slowest and is usually more useful for sleep or heavy drowsiness than normal studying. Theta is also slow. It can increase when a person is tired, but it can also appear during memory work and internal concentration. Alpha is often stronger during relaxed wakefulness, especially when the eyes are closed or the subject is calm. During demanding visual or mental work, alpha often becomes lower compared with rest. Beta is faster and is commonly linked with alertness, active thinking, and problem solving. Gamma is faster again, but for this first focus detector it is less practical because it can be noisy and can mix with muscle activity.

This is why the model should not depend on beta alone. Beta can rise during thinking, but also during tension or movement. Alpha can drop during attention, but lighting and eye behavior can also change it. Theta can be meaningful during memory work, but it can also rise when the subject gets sleepy. Ratios help because they compare bands with each other. Beta divided by alpha can show active thinking relative to relaxed wakefulness. Beta divided by theta can help separate alert effort from drifting or sleepy states. Theta divided by alpha adds context about relaxation and internal focus.

The most important part of this project is the labels. If a row is labeled focused, the subject should actually be doing a known focus task at that moment. If a row is labeled relaxed, the subject should be in a low-demand state. Guessing labels afterward is risky because the subject may change state quickly. The interface should collect only when the state is known. Reading gives a study-like focus state. Arithmetic gives a problem-solving focus state. A calm scenery stage gives a relaxed comparison state.

The reading task must be long enough that the subject cannot finish early. If the text is too short, the final part of the stage may be labeled focused reading even though the subject is just waiting. That would make the training data worse. A longer text keeps the subject engaged for the full timer. Questions after the reading are also useful because they make the subject treat the text seriously. The goal is not to grade the subject. The goal is to make the focus label more believable.

Arithmetic gives another useful focus state. Multiplication and multi-step calculations force the subject to hold numbers in working memory and update intermediate results. This is close to what students do when solving exercises. The answer correctness is useful metadata, but it is not the same thing as the EEG label. A wrong answer can still happen during real focus.

The relaxed condition should be low effort. A calm scene with slow movement is enough for early testing. Later the project can use open-license nature or animal videos, but the task should stay simple. It should not ask the subject to read, calculate, or memorize details.

A good first model should be personal and simple. EEG differs between people, electrodes, hair, movement, fatigue, and hardware settings. A model trained on one subject may not work for another. The first goal is not a universal focus detector. The first goal is to collect clean labeled data, train a small model, and check whether it performs better than chance on held-out samples.
""".strip(),
        "questions": [
            ("Why should labels come from the current task stage?", ["Because EEG state changes quickly", "Because labels are optional", "Because beta is always focus", "Because OpenBCI has no ports"], 0),
            ("Which bands are used by the first focus model?", ["Delta and gamma", "Theta, alpha, and beta", "Only beta", "All bands only"], 1),
            ("Why should the reading text be long?", ["To avoid finishing early while still recording focused data", "To remove the relaxed class", "To increase gamma", "To make CSV smaller"], 0),
            ("Why are beta/alpha and beta/theta useful?", ["They compare active thinking against context bands", "They remove all noise", "They replace labels", "They decode thoughts"], 0),
        ],
    },
    {
        "title": "Memory, attention, and working memory",
        "text": """
Memory, attention, and working memory

When students say they are focused, they usually mean that their attention is staying on one task for more than a few seconds. That sounds simple, but attention is not a single switch. The mind has to select useful information, ignore distractions, hold recent details in memory, and update those details when new information arrives. Reading a science paragraph, solving an equation, and comparing two definitions all require attention, but they stress the brain in slightly different ways.

Working memory is the short-term mental workspace used to hold and manipulate information. If a student multiplies 17 by 8, they may hold 10 times 8, 7 times 8, and the final sum in mind. If they read a paragraph, they hold the start of the sentence long enough to understand the end of the sentence. Working memory is limited. When the task becomes too difficult or when distractions appear, the student may lose the thread and need to restart.

Attention and working memory are closely connected. A person can look at text without processing it. The eyes move, but the meaning is not retained. In a study application, this matters because the visible behavior may look normal even when the mental state has changed. EEG cannot solve this perfectly, but it gives extra information that may help estimate whether the subject is in a study-like state or a relaxed state.

For this project, labels are created by controlled tasks. During a reading task, the subject has to process text and remember important ideas. During an arithmetic task, the subject has to maintain and update numbers. During a relaxed task, the subject watches calm scenery without needing to respond. These tasks do not guarantee perfect mental states, but they are much better than asking the model to guess labels after the fact.

A binary focus classifier should avoid making too many promises. It is not measuring intelligence, motivation, or whether the student understands the material. It is only estimating whether the current band-power pattern looks closer to the focused examples or the relaxed examples collected during training. This is why confidence is useful. A low-confidence prediction should be treated as uncertain instead of being shown as a strong claim.

Multiple subjects make the problem harder. People differ in resting alpha, eye movement, fatigue, electrode contact, and how they react to tasks. One person may show a clear beta increase during arithmetic, while another may show a stronger alpha decrease during reading. A training dataset should include several sessions and a balanced number of focused and relaxed samples. If possible, the same subject should also provide calibration data so the model can adjust to that person's baseline.

Good data collection is boring but important. The subject should sit still, avoid jaw movement, reduce blinking when possible, and keep the headset placement consistent. The room should be similar across sessions. A noisy session can harm the model more than a small model can fix. Clean labels and steady collection are more valuable than complicated algorithms.

The final system should therefore combine several pieces: a controlled interface for labeled collection, a simple feature set based on theta, alpha, and beta, a trained SVM model, and a frontend that receives only the final prediction and confidence. The less unnecessary information the frontend receives, the easier it is to build a clear student-facing application.
""".strip(),
        "questions": [
            ("What is working memory used for?", ["Holding and manipulating information briefly", "Controlling the OpenBCI port", "Removing all EEG noise", "Saving video files"], 0),
            ("Why are controlled tasks better than guessed labels?", ["They make the subject state known during collection", "They make calibration unnecessary", "They remove all artifacts", "They turn EEG into text"], 0),
            ("What should low confidence mean?", ["The prediction is uncertain", "The model is always wrong", "The subject is definitely relaxed", "The EEG headset is broken"], 0),
            ("Why do multiple subjects make the problem harder?", ["People differ in baseline rhythms and artifacts", "All people have identical EEG", "SVM cannot handle CSV files", "Theta disappears in groups"], 0),
        ],
    },
    {
        "title": "Why calibration matters",
        "text": """
Why calibration matters

A focus detector trained from EEG band powers has to deal with large differences between people. Even when two subjects perform the same task, their band powers may not look identical. One person may naturally show stronger alpha while relaxed. Another person may have more movement artifacts. A third person may have weaker signal quality because of hair, headset position, or electrode contact. Calibration helps because it gives the system a short example of what the current subject looks like before normal classification begins.

Calibration should not be confused with full training. Training creates the model by learning patterns from labeled examples. Calibration adjusts the incoming data relative to a user baseline. In this project, calibration can collect relaxed samples first. The backend can average theta, alpha, and beta during that relaxed state and then subtract the baseline from later incoming band powers. This gives the model information that is closer to how the current subject differs from their own normal relaxed state.

Calibration is useful, but it is not magic. If calibration is collected while the subject is moving, talking, blinking heavily, or already doing a mental task, the baseline will be poor. A bad baseline can make predictions worse. The subject should sit still and look at the calm screen during calibration. The calibration period should be long enough to average out short spikes, but not so long that the subject becomes bored or changes state dramatically.

The frontend should guide calibration clearly. It can show a countdown and ask the subject to relax. During that period, the frontend repeatedly sends band powers to the backend's calibration endpoint. After enough samples are collected, the frontend calls the finish endpoint. The backend stores the baseline in a small file so the same baseline can be reused later if appropriate.

After calibration, classification is still binary. The user interface does not need to know every internal feature. It only needs the prediction and confidence. The backend can keep the model details hidden. This separation makes the system easier to maintain: the machine learning code can change without forcing the frontend to understand every feature calculation.

Calibration also makes experiments easier to compare. If every subject starts with a relaxed baseline, the model can focus more on changes from that baseline. This does not remove the need for training data, but it helps reduce the effect of different absolute band-power levels between subjects. It is especially useful for a student application where the headset may be placed quickly and the user may not be an EEG expert.

A good calibration design should be simple. Start calibration, gather samples, finish calibration, then classify. If a user feels the prediction is strange, the frontend can offer recalibration. Recalibration should overwrite the old baseline only after a successful new calibration period. The system should avoid silently changing the baseline during normal studying because that could hide real changes in attention.

The goal is not to make a medical-grade EEG system. The goal is to create a practical focus-support tool for students. Calibration is one small step toward making the predictions more personal and less dependent on one global average.
""".strip(),
        "questions": [
            ("What is calibration used for in this project?", ["Adjusting incoming data relative to a user's baseline", "Replacing training completely", "Creating gamma waves", "Deleting relaxed samples"], 0),
            ("What can make calibration worse?", ["Movement or collecting the baseline during the wrong state", "Using a countdown", "Saving a baseline file", "Asking the subject to relax"], 0),
            ("What should the frontend receive after classification?", ["Only classification and confidence", "Every internal feature", "Raw headset bytes", "The full training dataset"], 0),
            ("When should recalibration be offered?", ["When predictions seem strange or the setup changes", "After every single packet", "Only before model training", "Never"], 0),
        ],
    },
    {
        "title": "Building useful study data",
        "text": """
Building useful study data

Machine learning projects often fail because the dataset does not match the real problem. A focus application for students should not be trained only on random numbers or perfect laboratory examples. It should collect data during tasks that look at least somewhat like studying. Reading, recall questions, and arithmetic are useful because they cover different kinds of mental effort. Reading requires sustained attention to meaning. Recall requires memory retrieval. Arithmetic requires working memory and step-by-step updating.

The relaxed class is just as important as the focused class. If the model only sees focused examples, it cannot learn what focused is not. A relaxed scene gives the model contrast. The subject should not have to solve anything, read anything, or remember anything during relaxation. They should simply watch a calm scene. The task should be boring in a controlled way, not stressful or distracting.

The dataset should also be balanced. If there are thousands of focused samples and only a few relaxed samples, the model may learn to predict focused too often. If there are too many relaxed samples, it may become too conservative. Balanced folders for training, testing, and validation make evaluation more honest. The training folder teaches the model. The testing folder checks performance during development. The validation folder gives a final check on data the model did not use for tuning.

Separating task metadata from labels keeps the model clean. The label should be simple: focused or relaxed. The task kind can still be saved separately as reading, quiz, math, or relax. This lets researchers later ask whether the model performs differently on reading compared with math, without turning every task into a separate class. Simple labels are easier to train and easier to explain.

The feature file should also stay simple. For the current SVM, theta, alpha, and beta are enough. Ratios can be calculated during training and prediction. Delta and gamma can remain in the raw file for auditing, but they do not need to be in the model-ready feature file. This keeps the pipeline consistent from collection to training to backend prediction.

Testing should be done carefully. If data from the same short session appears in both training and testing, the model may look better than it really is. A stronger test is to keep full sessions or full subjects separated between folders. For example, training can contain several sessions, testing can contain a different session, and validation can contain another different session. This checks whether the model learned a useful pattern instead of memorizing one recording.

The interface should support repeated collection from many people. Different reading texts reduce memorization and boredom. More questions make it harder for subjects to coast through the reading task. A better relaxation scene makes the relaxed condition feel natural without using copyrighted video. These improvements do not make the EEG perfect, but they make the dataset more reliable.

The best version of this project is built in small steps. First, verify packets. Second, collect clean labels. Third, train a simple model. Fourth, connect it to the frontend. Fifth, evaluate whether predictions are useful. If the model performs poorly, the correct response is to improve data quality before making the model more complicated.
""".strip(),
        "questions": [
            ("Why does the relaxed class matter?", ["It gives contrast so the model learns what focused is not", "It is optional noise", "It replaces reading", "It makes labels unnecessary"], 0),
            ("Why keep labels simple?", ["Task type can be metadata while labels stay focused/relaxed", "It prevents CSV export", "It removes all errors", "It makes gamma required"], 0),
            ("What is a stronger evaluation setup?", ["Keep sessions or subjects separated between splits", "Put the same rows in all folders", "Train and test on one packet", "Use only focused samples"], 0),
            ("Why add multiple reading texts?", ["To reduce memorization and boredom across subjects", "To increase packet size", "To remove calibration", "To avoid collecting relaxed data"], 0),
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
        self.reading_set = self.choose_reading_set(args.reading_index)
        self.reading_questions = self.reading_set["questions"][:]
        random.shuffle(self.reading_questions)
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
        self.relax_items = []
        self.stages = [
            Stage("Prepare", None, "instruction", args.prepare_seconds, False),
            Stage("Focused reading", FOCUSED_LABEL, "reading", args.focus_seconds, True),
            Stage("Reading recall", FOCUSED_LABEL, "quiz", args.quiz_seconds, True),
            Stage("Focused arithmetic", FOCUSED_LABEL, "math", args.math_seconds, True),
            Stage("Relaxed scenery", RELAXED_LABEL, "relax", args.relax_seconds, True),
            Stage("Finished", None, "finished", 0, False),
        ]
        self.root.title("NeuroFocus EEG Training Collector")
        self.root.geometry("1080x800")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.open_csvs()
        self.reader = UDPReader(args.recv_port, self.sample_queue, self.stop_event)
        self.reader.start()
        self.next_stage()
        self.root.after(100, self.tick)

    def choose_reading_set(self, reading_index):
        if reading_index is None:
            return random.choice(READING_SETS)
        if reading_index < 0 or reading_index >= len(READING_SETS):
            raise ValueError(f"reading-index must be between 0 and {len(READING_SETS) - 1}")
        return READING_SETS[reading_index]

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
            "reading_title", "stage_elapsed_seconds", "packet_time", "packet_unix_time", "packet_size_bytes", "decoded_float_count",
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
        titles = "\n".join(f"  {idx}: {reading['title']}" for idx, reading in enumerate(READING_SETS))
        text = (
            f"OpenBCI UDP port: {self.args.recv_port}\n"
            f"Raw: {self.raw_output_path}\n"
            f"Features: {self.features_output_path}\n"
            f"Feature bands: {', '.join(self.feature_bands)}\n"
            f"Labels: {FOCUSED_LABEL}, {RELAXED_LABEL}\n"
            f"Selected reading: {self.reading_set['title']}\n\n"
            "The reading is intentionally long. Keep reading until time ends; questions follow.\n\n"
            f"Available readings:\n{titles}"
        )
        ttk.Label(self.content, text=text, wraplength=980, font=("Arial", 15)).pack(anchor="nw")

    def show_reading(self):
        ttk.Label(
            self.content,
            text=f"{self.reading_set['title']} — keep reading until the timer ends. Recall questions come next.",
            font=("Arial", 15, "bold"),
            wraplength=980,
        ).pack(anchor="nw", pady=(0, 10))
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
        ttk.Label(self.content, text=f"Recall: {self.reading_set['title']}", wraplength=980, font=("Arial", 15, "bold")).pack(anchor="nw")
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

        # Distant forest layers
        for x in range(-40, width + 80, 55):
            self.canvas.create_polygon(x, 300, x + 35, 175, x + 70, 300, fill="#477f55", outline="")
        for x in range(-20, width + 80, 70):
            self.canvas.create_polygon(x, 330, x + 45, 190, x + 90, 330, fill="#2f6b46", outline="")
        for x in range(20, width, 120):
            self.canvas.create_rectangle(x + 38, 275, x + 52, 355, fill="#7a4d2b", outline="")
            self.canvas.create_oval(x, 220, x + 90, 305, fill="#2e7d4f", outline="")

        # River highlights
        for y in range(405, 485, 24):
            self.canvas.create_arc(0, y, width, y + 70, start=5, extent=40, style="arc", outline="#bce9ff", width=2)

        self.relax_items = []
        for _ in range(7):
            x, y = random.randint(0, width - 110), random.randint(35, 140)
            cloud_parts = [
                self.canvas.create_oval(x, y, x + 70, y + 35, fill="white", outline=""),
                self.canvas.create_oval(x + 35, y - 12, x + 105, y + 35, fill="white", outline=""),
            ]
            self.relax_items.append({"items": cloud_parts, "dx": random.uniform(0.25, 0.7), "wrap": width + 140})

        for _ in range(5):
            x, y = random.randint(0, width - 60), random.randint(120, 220)
            bird = [
                self.canvas.create_arc(x, y, x + 28, y + 20, start=20, extent=140, style="arc", outline="#263238", width=2),
                self.canvas.create_arc(x + 24, y, x + 52, y + 20, start=20, extent=140, style="arc", outline="#263238", width=2),
            ]
            self.relax_items.append({"items": bird, "dx": random.uniform(0.8, 1.8), "wrap": width + 80})

        for _ in range(12):
            x, y = random.randint(0, width - 20), random.randint(335, 380)
            leaf = self.canvas.create_oval(x, y, x + 14, y + 7, fill=random.choice(["#d6a84f", "#9fbe5a", "#c46b3b"]), outline="")
            self.relax_items.append({"items": [leaf], "dx": random.uniform(0.35, 0.9), "dy": random.uniform(-0.08, 0.08), "wrap": width + 40})

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
            dx = obj.get("dx", 0.5)
            dy = obj.get("dy", 0.0)
            for item in obj["items"]:
                self.canvas.move(item, dx, dy)
            bbox = self.canvas.bbox(obj["items"][0])
            if bbox and bbox[0] > obj.get("wrap", width + 100):
                for item in obj["items"]:
                    self.canvas.move(item, -width - 140, 0)
        self.root.after(40, self.animate_relax_scene)

    def show_finished(self):
        ttk.Label(self.content, text=f"Session finished.\nSaved labeled samples: {self.labeled_sample_count}\nRaw: {self.raw_output_path}\nFeatures: {self.features_output_path}", wraplength=900, font=("Arial", 16)).pack(anchor="nw")

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
                "reading_title": self.reading_set["title"] if stage.kind in {"reading", "quiz"} else "",
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
    parser.add_argument("--reading-index", type=int, default=None, help="Select a fixed reading set index. Defaults to random.")
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
