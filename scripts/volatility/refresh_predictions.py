"""
refresh_predictions.py

The DAILY automated job (see .github/workflows/daily-refresh.yml). Pulls
fresh price data, computes TODAY's features for each ticker, and runs
them through the ALREADY-TRAINED model to update the live prediction --
it does NOT retrain anything. See weekly_retrain.yml /
promote_if_safe.py for the separate, slower, human-reviewed retrain
cadence, and the README's "Keeping this up to date" section for why
these are deliberately two different jobs.

WHY A SHORT FETCH WINDOW: the full pipeline (01_fetch_data.py) pulls
back to 2005 because training needs that much history. This job only
needs enough trailing days to compute ONE valid row of rolling features
(the 20-day realized-vol window is the longest one) -- FETCH_WINDOW_DAYS
is generous padding above that, not an attempt to re-derive the whole
dataset daily.

WHY THIS DOESN'T RETRAIN: see the module docstring in
05_train_model.py / common.py's select_f1_thresholds_per_ticker --
this project already found that an unreviewed automatic process can
silently ship a regression (the per-ticker threshold bug). A daily
job that only computes NEW inputs through a FIXED, already-reviewed
model can't reintroduce that kind of bug; a daily job that retrains
unattended could.

Usage:
    python scripts/volatility/refresh_predictions.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (
    BASKET, MODELS_DIR, apply_per_ticker_thresholds,
    compute_shap_explanations, engineer_single_ticker, top_features_for_row,
)
# Deliberately NOT importing common.build_feature_matrix: this script
# builds a feature matrix for INFERENCE on already-engineered live rows
# (numeric cols come from metadata.json, matching 06_evaluate.py's
# style), not for training. common.build_feature_matrix has a different
# signature/return shape (it also returns the inferred dummy columns,
# used only when there's no metadata yet to read them from) -- see the
# local build_feature_matrix() below.
from data_utils import VIX_SYMBOL, fetch_ticker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_DATA_DIR = PROJECT_ROOT / "backend" / "data" / "volatility"

FETCH_WINDOW_DAYS = 90  # calendar days of padding above the longest
                          # rolling window (20 trading days) used by any
                          # feature -- generous on purpose so holidays/
                          # weekends can never leave the latest row with
                          # NaN features.


def fetch_latest_window() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    start = (datetime.today() - timedelta(days=FETCH_WINDOW_DAYS)).strftime("%Y-%m-%d")
    raw = {}
    for ticker in BASKET:
        raw[ticker] = fetch_ticker(ticker, start, end=None)
    vix = fetch_ticker(VIX_SYMBOL, start, end=None)
    return raw, vix


def build_latest_rows(raw: dict[str, pd.DataFrame], vix: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker -- the most recent trading day's features."""
    rows = []
    for ticker in BASKET:
        engineered = engineer_single_ticker(raw[ticker], vix, ticker)
        engineered = engineered.dropna()
        if engineered.empty:
            print(f"WARNING: {ticker} has no valid feature row after the "
                  f"{FETCH_WINDOW_DAYS}-day fetch window -- skipping it "
                  "today rather than writing a prediction built on NaNs.",
                  file=sys.stderr)
            continue
        rows.append(engineered.iloc[[-1]])  # most recent row only
    return pd.concat(rows)


def load_model_artifacts():
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        print("ERROR: models/volatility/metadata.json not found. Run the "
              "full training pipeline (05_train_model.py, 06_evaluate.py) "
              "at least once before this job can run.", file=sys.stderr)
        sys.exit(1)
    with open(meta_path) as f:
        metadata = json.load(f)
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    return metadata, xgb


def update_latest_predictions(
    latest_rows: pd.DataFrame, xgb, metadata: dict
) -> dict:
    numeric_cols = metadata["numeric_feature_cols"]
    ticker_dummy_cols = metadata["ticker_dummy_cols"]
    feature_cols = metadata["feature_cols"]

    X = build_feature_matrix(latest_rows, numeric_cols, ticker_dummy_cols)
    proba = xgb.predict_proba(X)[:, 1]

    thresholds = metadata["decision_threshold"]["xgboost"]
    if not isinstance(thresholds, dict):  # backward compatibility
        thresholds = {t: thresholds for t in metadata["basket"]}
    pred = apply_per_ticker_thresholds(proba, latest_rows["ticker"], thresholds)

    _, shap_values, _ = compute_shap_explanations(xgb, X, feature_cols)

    predictions = {}
    for i, (date, row) in enumerate(latest_rows.iterrows()):
        ticker = row["ticker"]
        regime = "high" if pred[i] == 1 else "low"
        confidence = proba[i] if regime == "high" else 1 - proba[i]
        predictions[ticker] = {
            "date": date.strftime("%Y-%m-%d"),
            "regime": regime,
            "probability": round(float(confidence), 4),
            "top_features": top_features_for_row(shap_values[i], feature_cols),
            "model": "xgboost",
            "note": None,
        }
    return predictions


def build_feature_matrix(df: pd.DataFrame, numeric_cols: list, ticker_dummy_cols: list):
    numeric = df[numeric_cols]
    dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    dummies = dummies.reindex(columns=ticker_dummy_cols, fill_value=0)
    return pd.concat([numeric, dummies], axis=1)


def append_to_histories(predictions: dict, latest_rows: pd.DataFrame) -> dict:
    path = BACKEND_DATA_DIR / "histories.json"
    histories = json.loads(path.read_text()) if path.exists() else {}

    price_by_ticker = {row["ticker"]: row["Close"] for _, row in latest_rows.iterrows()}

    for ticker, pred in predictions.items():
        histories.setdefault(ticker, {"dates": [], "prices": [], "regimes": []})
        h = histories[ticker]
        if pred["date"] in h["dates"]:
            # Already have this date (job re-run same day, or no new
            # trading day yet) -- update in place instead of duplicating.
            idx = h["dates"].index(pred["date"])
            h["prices"][idx] = round(float(price_by_ticker[ticker]), 2)
            h["regimes"][idx] = pred["regime"]
        else:
            h["dates"].append(pred["date"])
            h["prices"].append(round(float(price_by_ticker[ticker]), 2))
            h["regimes"].append(pred["regime"])
    return histories


def main():
    print(f"Refreshing volatility predictions for: {', '.join(BASKET)}")

    metadata, xgb = load_model_artifacts()
    raw, vix = fetch_latest_window()
    latest_rows = build_latest_rows(raw, vix)

    if latest_rows.empty:
        print("ERROR: no ticker produced a valid latest row -- aborting "
              "without writing anything (keeping yesterday's predictions live "
              "rather than overwriting them with nothing).", file=sys.stderr)
        sys.exit(1)

    predictions = update_latest_predictions(latest_rows, xgb, metadata)
    histories = append_to_histories(predictions, latest_rows)

    BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(BACKEND_DATA_DIR / "latest_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    with open(BACKEND_DATA_DIR / "histories.json", "w") as f:
        json.dump(histories, f, indent=2)

    for ticker, pred in predictions.items():
        print(f"  {ticker:>4} ({pred['date']}): {pred['regime']:>4} vol, "
              f"{pred['probability']:.2%} confidence")
    print("\nDone.")


if __name__ == "__main__":
    main()
