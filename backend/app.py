"""
app.py -- Flask API for the SPY/QQQ/IWM/DIA/XLF Volatility Regime Predictor.

Serves per-ticker model predictions and history to the React frontend.

Data source:
- backend/data/latest_predictions.json and backend/data/histories.json
  are written by scripts/06_evaluate.py, each keyed by ticker (e.g.
  {"SPY": {...}, "QQQ": {...}, ...}). This file is a thin read-and-serve
  layer -- it does NOT run the model itself. Re-run 06_evaluate.py to
  update what the API serves; no need to restart Flask, files are read
  fresh on every request.

Notes carried over from earlier builds:
- Runs on port 5001, not 5000 (macOS AirPlay conflict).
- CORS handled manually via after_request (flask-cors was unreliable).
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

BACKEND_DATA_DIR = Path(__file__).resolve().parent / "data"
LATEST_PREDICTIONS_PATH = BACKEND_DATA_DIR / "latest_predictions.json"
HISTORIES_PATH = BACKEND_DATA_DIR / "histories.json"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF"]
DEFAULT_TICKER = "SPY"

FRONTEND_ORIGIN = "http://localhost:5173"


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = FRONTEND_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "volatility-predictor-api"})


@app.route("/api/tickers", methods=["GET"])
def tickers():
    """Which tickers the frontend can offer in its selector."""
    predictions = _load_json(LATEST_PREDICTIONS_PATH)
    available = list(predictions.keys()) if predictions else BASKET
    return jsonify({"tickers": available, "default": DEFAULT_TICKER})


@app.route("/api/predict/latest", methods=["GET"])
def predict_latest():
    """
    Returns the latest prediction for one ticker (?ticker=SPY, defaults
    to SPY). Read from backend/data/latest_predictions.json.
    """
    ticker = request.args.get("ticker", DEFAULT_TICKER).upper()
    predictions = _load_json(LATEST_PREDICTIONS_PATH)

    if ticker in predictions:
        return jsonify(predictions[ticker])

    return jsonify({
        "date": None, "regime": None, "probability": None,
        "top_features": [], "model": None,
        "note": f"No prediction available for {ticker} yet -- run the "
                f"pipeline (scripts 01-06) to populate this.",
    })


@app.route("/api/history", methods=["GET"])
def history():
    """Returns price + regime history for one ticker (?ticker=SPY)."""
    ticker = request.args.get("ticker", DEFAULT_TICKER).upper()
    histories = _load_json(HISTORIES_PATH)

    if ticker in histories:
        return jsonify(histories[ticker])

    return jsonify({"dates": [], "prices": [], "regimes": []})


if __name__ == "__main__":
    app.run(port=5001, debug=True)
