#!/usr/bin/env python3
"""Local frontend/backend bridge for EEG focus classification.

This service loads a saved SVM model and exposes a small HTTP API for the
frontend. The frontend sends OpenBCI band powers, the backend extracts the same
features used during training, runs the saved model, and returns the focus
classification plus confidence.

Run:
    python focus_backend_server.py --model focus_svm.joblib --host 127.0.0.1 --port 8000

Example request:
    POST http://127.0.0.1:8000/classify
    Content-Type: application/json

    {
      "theta": 0.671393,
      "alpha": 0.166967,
      "beta": 0.104697
    }

Also accepted:
    {"band_powers": {"delta": 0.03, "theta": 0.67, "alpha": 0.16, "beta": 0.10, "gamma": 0.01}}
    {"band_powers": [delta, theta, alpha, beta, gamma]}
    {"band_powers": [theta, alpha, beta]}

Response:
    {
      "classification": "focused",
      "confidence": 0.83,
      "features": {...}
    }
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import joblib
import pandas as pd

from eeg_svm_pipeline import MODEL_FEATURES, add_ratio_features, prepare_features

OPENBCI_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
REQUIRED_BANDS = ["theta", "alpha", "beta"]
DEFAULT_LABEL_MAP = {1: "focused", 0: "relaxed"}


class FocusClassifier:
    """Thin wrapper around the saved sklearn model."""

    def __init__(self, model_path: str, threshold: float = 0.5):
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.threshold = threshold

    def classify(self, payload: dict[str, Any]) -> dict[str, Any]:
        bands = extract_band_payload(payload)
        row = pd.DataFrame([bands])
        row = add_ratio_features(row)
        X = prepare_features(
            row[MODEL_FEATURES],
            z_thresh=3.5,
            smooth_window=1,
            fill_method="median",
        )

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
            focused_probability = None

        classification = DEFAULT_LABEL_MAP.get(predicted_class, str(predicted_class))

        return {
            "classification": classification,
            "confidence": confidence,
            "focused_probability": focused_probability,
            "threshold": self.threshold,
            "features": {feature: float(X.iloc[0][feature]) for feature in MODEL_FEATURES},
        }


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
        server_version = "NeuroFocusBackend/1.0"

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

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model_path": classifier.model_path,
                        "required_bands": REQUIRED_BANDS,
                        "features": MODEL_FEATURES,
                    },
                )
                return

            self._send_json(
                404,
                {
                    "error": "Not found",
                    "available_endpoints": ["GET /health", "POST /classify"],
                },
            )

        def do_POST(self) -> None:
            if self.path != "/classify":
                self._send_json(404, {"error": "Not found", "available_endpoints": ["POST /classify"]})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                result = classifier.classify(payload)
                self._send_json(200, result)
            except Exception as exc:
                logging.exception("Failed to classify request")
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    classifier = FocusClassifier(args.model, threshold=args.threshold)
    handler = make_handler(classifier)
    server = ThreadingHTTPServer((args.host, args.port), handler)

    logging.info("NeuroFocus backend listening on http://%s:%s", args.host, args.port)
    logging.info("Health check: http://%s:%s/health", args.host, args.port)
    logging.info("Classify endpoint: POST http://%s:%s/classify", args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping backend")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
