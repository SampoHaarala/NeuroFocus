#!/usr/bin/env python3
"""Local frontend/backend bridge for EEG focus classification.

The frontend sends OpenBCI band powers. The backend extracts the same internal
features used during SVM training, optionally applies a per-user calibration
baseline, runs the saved model, and returns only binary classification and
confidence.

Run:
    python focus_backend_server.py --model focus_svm.joblib --host 127.0.0.1 --port 8000

Main endpoints:
    GET  /health
    POST /classify
    POST /calibration/start
    POST /calibration/sample
    POST /calibration/finish
    GET  /calibration/status
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from eeg_svm_pipeline import MODEL_FEATURES, add_ratio_features, prepare_features

OPENBCI_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
REQUIRED_BANDS = ["theta", "alpha", "beta"]
DEFAULT_LABEL_MAP = {1: "focused", 0: "relaxed"}


class CalibrationStore:
    """Stores a simple per-user resting baseline for incoming band powers."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.active_samples: list[dict[str, float]] = []
        self.baseline: dict[str, float] | None = None
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            baseline = data.get("baseline")
            if isinstance(baseline, dict):
                self.baseline = {band: float(baseline[band]) for band in REQUIRED_BANDS if band in baseline}
        except Exception:
            logging.exception("Could not load calibration file")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"baseline": self.baseline}, indent=2), encoding="utf-8")

    def start(self) -> None:
        self.active_samples = []

    def add_sample(self, bands: dict[str, float]) -> int:
        self.active_samples.append({band: float(bands[band]) for band in REQUIRED_BANDS})
        return len(self.active_samples)

    def finish(self) -> dict[str, Any]:
        if not self.active_samples:
            raise ValueError("No calibration samples collected")
        self.baseline = {
            band: sum(sample[band] for sample in self.active_samples) / len(self.active_samples)
            for band in REQUIRED_BANDS
        }
        count = len(self.active_samples)
        self.active_samples = []
        self.save()
        return {"calibrated": True, "samples": count}

    def apply(self, bands: dict[str, float]) -> dict[str, float]:
        if not self.baseline:
            return bands
        # Express current band powers relative to this user's relaxed baseline.
        # This keeps the model input shape the same while reducing user-to-user offset.
        adjusted = {}
        for band in REQUIRED_BANDS:
            baseline_value = self.baseline.get(band, 0.0)
            adjusted[band] = float(bands[band]) - float(baseline_value)
        return adjusted

    def status(self) -> dict[str, Any]:
        return {
            "calibrated": self.baseline is not None,
            "collecting_samples": len(self.active_samples),
        }


class FocusClassifier:
    """Thin wrapper around the saved sklearn model."""

    def __init__(self, model_path: str, threshold: float = 0.5, calibration_path: str = "calibration.json"):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.threshold = threshold
        self.calibration = CalibrationStore(calibration_path)

    def classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        bands = extract_band_payload(payload)
        calibrated_bands = self.calibration.apply(bands)
        X = build_model_input(calibrated_bands)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            classes = list(self.model.classes_)
            if 1 in classes:
                focused_probability = float(proba[classes.index(1)])
            else:
                focused_probability = float(max(proba))
            predicted_class = 1 if focused_probability >= self.threshold else 0
            confidence = focused_probability if predicted_class == 1 else 1.0 - focused_probability
        else:
            predicted_class = int(self.model.predict(X)[0])
            confidence = None

        return {
            "classification": DEFAULT_LABEL_MAP.get(predicted_class, str(predicted_class)),
            "confidence": confidence,
        }


def build_model_input(bands: dict[str, float]) -> pd.DataFrame:
    row = pd.DataFrame([bands])
    row = add_ratio_features(row)
    return prepare_features(
        row[MODEL_FEATURES],
        z_thresh=3.5,
        smooth_window=1,
        fill_method="median",
    )


def extract_band_payload(payload: dict[str, Any]) -> dict[str, float]:
    """Normalize incoming frontend payloads into theta/alpha/beta values."""
    source = payload.get("band_powers", payload)

    if isinstance(source, dict):
        normalized = {str(key).strip().lower(): value for key, value in source.items()}
        missing = [band for band in REQUIRED_BANDS if band not in normalized]
        if missing:
            raise ValueError(f"Missing required band power values: {missing}")
        return {band: float(normalized[band]) for band in REQUIRED_BANDS}

    if isinstance(source, list):
        if len(source) == 5:
            mapped = dict(zip(OPENBCI_BANDS, source))
        elif len(source) == 3:
            mapped = dict(zip(REQUIRED_BANDS, source))
        else:
            raise ValueError(
                "band_powers list must contain either 3 values [theta, alpha, beta] "
                "or 5 OpenBCI values [delta, theta, alpha, beta, gamma]"
            )
        return {band: float(mapped[band]) for band in REQUIRED_BANDS}

    raise ValueError("Expected JSON object with theta/alpha/beta or a band_powers object/list")


def make_handler(classifier: FocusClassifier):
    class FocusRequestHandler(BaseHTTPRequestHandler):
        server_version = "NeuroFocusBackend/1.1"

        def _send_json(self, status_code: int, body: dict[str, Any]) -> None:
            response = json.dumps(body).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(response)

        def _read_json_body(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            return json.loads(raw_body.decode("utf-8")) if raw_body else {}

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            if self.path == "/calibration/status":
                self._send_json(200, classifier.calibration.status())
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            try:
                if self.path == "/classify":
                    result = classifier.classify(self._read_json_body())
                    self._send_json(200, result)
                    return

                if self.path == "/calibration/start":
                    classifier.calibration.start()
                    self._send_json(200, {"calibrating": True, "samples": 0})
                    return

                if self.path == "/calibration/sample":
                    bands = extract_band_payload(self._read_json_body())
                    count = classifier.calibration.add_sample(bands)
                    self._send_json(200, {"calibrating": True, "samples": count})
                    return

                if self.path == "/calibration/finish":
                    result = classifier.calibration.finish()
                    self._send_json(200, result)
                    return

                self._send_json(404, {"error": "Not found"})
            except Exception as exc:
                logging.exception("Request failed")
                self._send_json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            logging.info("%s - %s", self.address_string(), format % args)

    return FocusRequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local EEG focus classification backend")
    parser.add_argument("--model", required=True, help="Path to saved sklearn/joblib SVM model")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port. Default: 8000")
    parser.add_argument("--threshold", type=float, default=0.5, help="Focused probability threshold. Default: 0.5")
    parser.add_argument("--calibration-file", default="calibration.json", help="Where to store user calibration baseline")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    classifier = FocusClassifier(
        args.model,
        threshold=args.threshold,
        calibration_path=args.calibration_file,
    )
    handler = make_handler(classifier)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    logging.info("NeuroFocus backend listening on http://%s:%s", args.host, args.port)
    logging.info("Classify endpoint: POST http://%s:%s/classify", args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping backend")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
