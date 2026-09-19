"""
app.py -- Flask API for the SPY Volatility Regime Predictor.

Serves model predictions and supporting data (feature values, SHAP
explanations) to the React frontend.

Notes carried over from the Startup Predictor build:
- Runs on port 5001, not 5000, to avoid colliding with the macOS
  AirPlay Receiver service that squats on 5000.
- CORS is handled manually via after_request rather than flask-cors,
  which was unreliable in the last project.

Data source:
- backend/data/latest_prediction.json and backend/data/history.json are
  written by scripts/06_evaluate.py. This file is a thin read-and-serve
  layer -- it does NOT run the model itself. Re-run 06_evaluate.py
  (which re-runs 05's models against fresh data) to update what the API
  serves; there's no need to restart Flask afterward since the files
  are read fresh on every request.
"""

import json
from pathlib import Path

from flask import Flask, jsonify

app = Flask(__name__)

BACKEND_DATA_DIR = Path(__file__).resolve().parent / "data"
LATEST_PREDICTION_PATH = BACKEND_DATA_DIR / "latest_prediction.json"
HISTORY_PATH = BACKEND_DATA_DIR / "history.json"

# --- CORS (manual, per lesson learned on the last project) ---------------
FRONTEND_ORIGIN = "http://localhost:5173"  # Vite's default dev server port


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _load_json_or_placeholder(path: Path, placeholder: dict) -> dict:
    """
    Reads a JSON artifact if it exists; otherwise returns a clearly
    labeled placeholder rather than erroring, so the frontend keeps
    working (showing "not available yet") even before the pipeline has
    been run end-to-end.
    """
    if not path.exists():
        return placeholder
    with open(path) as f:
        return json.load(f)


# --- Routes ----------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Simple liveness check -- confirms the API is up and reachable."""
    return jsonify({"status": "ok", "service": "spy-volatility-predictor-api"})


@app.route("/api/predict/latest", methods=["GET"])
def predict_latest():
    """
    Returns the model's most recent volatility regime prediction, plus
    its SHAP-based explanation. Read from backend/data/latest_prediction.json,
    written by scripts/06_evaluate.py.
    """
    placeholder = {
        "date": None,
        "regime": None,
        "probability": None,
        "top_features": [],
        "model": None,
        "note": "Model not yet trained -- run the pipeline (scripts 01-06) "
                "to populate this.",
    }
    return jsonify(_load_json_or_placeholder(LATEST_PREDICTION_PATH, placeholder))


@app.route("/api/history", methods=["GET"])
def history():
    """
    Returns historical SPY price + predicted regime data (test-set
    period) for the frontend's chart. Read from
    backend/data/history.json, written by scripts/06_evaluate.py.
    """
    placeholder = {"dates": [], "prices": [], "regimes": []}
    return jsonify(_load_json_or_placeholder(HISTORY_PATH, placeholder))


if __name__ == "__main__":
    app.run(port=5001, debug=True)
