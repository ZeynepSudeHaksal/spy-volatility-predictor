"""
06_evaluate.py

Final, honest evaluation on the test split -- pooled AND per ticker.
Also runs SHAP on XGBoost and writes the per-ticker prediction/history
artifacts the Flask backend serves. Same structure as the companion
volatility project's 06_evaluate.py.

THE HONEST HEADLINE NUMBER FOR THIS PROJECT
------------------------------------------------
Report test accuracy ALONGSIDE the always-up baseline from training,
every time -- never accuracy alone. A model at 56% sounds unimpressive
in isolation and highly impressive next to a 51% baseline, or
unimpressive next to a 58% one. The gap over baseline, not the raw
number, is what this project is actually trying to demonstrate.

Usage:
    python scripts/06_evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "direction"
MODELS_DIR = PROJECT_ROOT / "models" / "direction"
BACKEND_DATA_DIR = PROJECT_ROOT / "backend" / "data" / "direction"


def load_artifacts():
    test_path = PROCESSED_DATA_DIR / "test.csv"
    meta_path = MODELS_DIR / "metadata.json"
    if not test_path.exists():
        print("ERROR: test.csv not found. Run 04_train_test_split.py first.", file=sys.stderr)
        sys.exit(1)
    if not meta_path.exists():
        print("ERROR: models/metadata.json not found. Run 05_train_model.py first.", file=sys.stderr)
        sys.exit(1)

    test = pd.read_csv(test_path, index_col="Date", parse_dates=True)
    with open(meta_path) as f:
        metadata = json.load(f)
    log_reg = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    return test, metadata, log_reg, scaler, xgb


def build_feature_matrix(df, numeric_cols, ticker_dummy_cols):
    numeric = df[numeric_cols]
    dummies = pd.get_dummies(df["ticker"], prefix="ticker").reindex(
        columns=ticker_dummy_cols, fill_value=0
    )
    return pd.concat([numeric, dummies], axis=1)


def evaluate_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
    }


def top_features_for_row(shap_row, feature_cols, n=5) -> list:
    ranked = sorted(zip(feature_cols, shap_row), key=lambda x: -abs(x[1]))[:n]
    return [
        {"feature": feat, "shap_value": round(float(val), 4)}
        for feat, val in ranked
        if not feat.startswith("ticker_")
    ]


def main():
    test, metadata, log_reg, scaler, xgb = load_artifacts()
    numeric_cols = metadata["numeric_feature_cols"]
    ticker_dummy_cols = metadata["ticker_dummy_cols"]
    feature_cols = metadata["feature_cols"]
    basket = metadata["basket"]
    always_up_baseline = metadata["always_up_baseline_val_accuracy"]

    X_test = build_feature_matrix(test, numeric_cols, ticker_dummy_cols)
    y_test = test["label"]

    test_always_up_acc = accuracy_score(y_test, np.ones(len(y_test), dtype=int))
    print(f"Always-up baseline -- TEST: {test_always_up_acc:.4f} "
          f"(val was {always_up_baseline:.4f})")

    X_test_scaled = scaler.transform(X_test)
    lr_pred = log_reg.predict(X_test_scaled)
    lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    lr_pooled = evaluate_metrics(y_test, lr_pred, lr_proba)
    print(f"\nLogistic Regression -- TEST (pooled): accuracy={lr_pooled['accuracy']:.4f} "
          f"({'beats' if lr_pooled['accuracy']>test_always_up_acc else 'does NOT beat'} baseline), "
          f"roc_auc={lr_pooled['roc_auc']:.4f}")

    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_pooled = evaluate_metrics(y_test, xgb_pred, xgb_proba)
    print(f"XGBoost -- TEST (pooled): accuracy={xgb_pooled['accuracy']:.4f} "
          f"({'beats' if xgb_pooled['accuracy']>test_always_up_acc else 'does NOT beat'} baseline), "
          f"roc_auc={xgb_pooled['roc_auc']:.4f}")

    print("\nXGBoost -- TEST, per ticker (vs. that ticker's own always-up rate):")
    xgb_per_ticker = {}
    for ticker in basket:
        mask = (test["ticker"] == ticker).values
        if mask.sum() == 0:
            continue
        m = evaluate_metrics(y_test[mask], xgb_pred[mask], xgb_proba[mask])
        ticker_always_up = float(y_test[mask].mean())
        m["always_up_baseline"] = ticker_always_up
        xgb_per_ticker[ticker] = m
        beats = "beats" if m["accuracy"] > ticker_always_up else "does NOT beat"
        print(f"  {ticker:>4}: accuracy={m['accuracy']:.3f} vs always-up={ticker_always_up:.3f} "
              f"({beats})  roc_auc={m['roc_auc']:.3f}")

    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    global_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    print("\nSHAP global feature importance (mean |SHAP value|, pooled):")
    for feat, val in global_importance.items():
        print(f"  {feat:>18}: {val:.4f}")

    production_model_name = (
        "xgboost" if xgb_pooled["roc_auc"] >= lr_pooled["roc_auc"] else "logistic_regression"
    )
    print(f"\nProduction model selected: {production_model_name}")

    BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    latest_predictions, histories = {}, {}
    for ticker in basket:
        mask = (test["ticker"] == ticker).values
        positions = np.where(mask)[0]
        if len(positions) == 0:
            continue
        last_pos = positions[-1]
        last_date = test.index[last_pos]

        # xgb_proba is P(up). "probability" here means "confidence in
        # the predicted direction" -- when the call is "down", that's
        # P(down) = 1 - proba, not the raw P(up), or a confident "down"
        # call would render as a low-looking probability next to a
        # "down expected" label.
        direction = "up" if xgb_pred[last_pos] == 1 else "down"
        confidence = xgb_proba[last_pos] if direction == "up" else 1 - xgb_proba[last_pos]

        latest_predictions[ticker] = {
            "date": last_date.strftime("%Y-%m-%d"),
            "direction": direction,
            "probability": round(float(confidence), 4),
            "always_up_baseline": round(float(y_test[mask].mean()), 4),
            "top_features": top_features_for_row(shap_values[last_pos], feature_cols),
            "model": "xgboost",
            "note": None,
        }
        histories[ticker] = {
            "dates": [d.strftime("%Y-%m-%d") for d in test.index[positions]],
            "prices": [round(float(p), 2) for p in test["Close"].values[positions]],
            "directions": ["up" if xgb_pred[p] == 1 else "down" for p in positions],
        }

    with open(BACKEND_DATA_DIR / "latest_predictions.json", "w") as f:
        json.dump(latest_predictions, f, indent=2)
    with open(BACKEND_DATA_DIR / "histories.json", "w") as f:
        json.dump(histories, f, indent=2)

    final_metadata = {
        **metadata,
        "test_always_up_baseline": test_always_up_acc,
        "test_metrics_pooled": {"logistic_regression": lr_pooled, "xgboost": xgb_pooled},
        "test_metrics_by_ticker": xgb_per_ticker,
        "shap_global_importance": global_importance.round(4).to_dict(),
        "production_model": production_model_name,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(final_metadata, f, indent=2)

    print(f"\nSaved backend artifacts -> {BACKEND_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
