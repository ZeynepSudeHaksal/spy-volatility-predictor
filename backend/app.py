"""
app.py -- Flask API for the Market Analytics dashboard.

Serves TWO prediction modes from one app, each fully independent:

- Volatility regime (scripts/volatility/ pipeline -> backend/data/volatility/)
- Direction (up/down)  (scripts/direction/ pipeline -> backend/data/direction/)

Both read the same basket of tickers (SPY, QQQ, IWM, DIA, XLF, GLD,
TLT), fetched once by scripts/01_fetch_data.py, but are otherwise
separate models with
separate features, labels, and metrics -- routed under /api/volatility/
and /api/direction/ respectively so they can evolve independently
without colliding.

Notes carried over from earlier builds:
- Runs on port 5001 (macOS AirPlay Receiver squats on 5000).
- CORS handled manually via after_request (flask-cors was unreliable).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

BACKEND_DATA_DIR = Path(__file__).resolve().parent / "data"
BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF", "GLD", "TLT"]
DEFAULT_TICKER = "SPY"

# Configurable via env var so the same image works in dev (Vite on :5173)
# and in a container/deployment (whatever origin the frontend actually
# gets served from), without a code change.
FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")


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


def _register_mode_routes(mode: str, empty_prediction: dict, empty_history: dict):
    """
    Registers /api/<mode>/health, /api/<mode>/tickers,
    /api/<mode>/predict/latest, and /api/<mode>/history, all reading
    from backend/data/<mode>/. One function, called once per mode, so
    the two modes can't drift out of sync in how they're wired up.
    """
    mode_dir = BACKEND_DATA_DIR / mode
    latest_path = mode_dir / "latest_predictions.json"
    history_path = mode_dir / "histories.json"

    @app.route(f"/api/{mode}/health", methods=["GET"], endpoint=f"{mode}_health")
    def health():
        return jsonify({"status": "ok", "mode": mode})

    @app.route(f"/api/{mode}/tickers", methods=["GET"], endpoint=f"{mode}_tickers")
    def tickers():
        predictions = _load_json(latest_path)
        available = list(predictions.keys()) if predictions else BASKET
        return jsonify({"tickers": available, "default": DEFAULT_TICKER})

    @app.route(f"/api/{mode}/predict/latest", methods=["GET"], endpoint=f"{mode}_predict_latest")
    def predict_latest():
        ticker = request.args.get("ticker", DEFAULT_TICKER).upper()
        predictions = _load_json(latest_path)
        if ticker in predictions:
            return jsonify(predictions[ticker])
        return jsonify({
            **empty_prediction,
            "note": f"No prediction available for {ticker} yet -- run the "
                    f"{mode} pipeline (scripts/{mode}/02-06) to populate this.",
        })

    @app.route(f"/api/{mode}/history", methods=["GET"], endpoint=f"{mode}_history")
    def history():
        ticker = request.args.get("ticker", DEFAULT_TICKER).upper()
        histories = _load_json(history_path)
        if ticker in histories:
            return jsonify(histories[ticker])
        return jsonify(empty_history)


_register_mode_routes(
    "volatility",
    empty_prediction={
        "date": None, "regime": None, "probability": None,
        "top_features": [], "model": None, "note": None,
    },
    empty_history={"dates": [], "prices": [], "regimes": []},
)

_register_mode_routes(
    "direction",
    empty_prediction={
        "date": None, "direction": None, "probability": None,
        "always_up_baseline": None, "top_features": [], "model": None, "note": None,
    },
    empty_history={"dates": [], "prices": [], "directions": []},
)


@app.route("/api/health", methods=["GET"])
def root_health():
    return jsonify({"status": "ok", "service": "market-analytics-api", "modes": ["volatility", "direction"]})


if __name__ == "__main__":
    # host 0.0.0.0 so the Flask dev server is reachable from outside its
    # own container; debug stays opt-in via FLASK_DEBUG so it's off by
    # default (debug=True runs an auto-reloading dev server -- fine on a
    # laptop, not what you want in a container).
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=debug)
