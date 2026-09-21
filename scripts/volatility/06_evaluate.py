"""
06_evaluate.py

Final, honest evaluation on the test split, for the pooled basket
model -- both overall and broken down PER TICKER (a pooled model could
look good on average while quietly underperforming on one ticker, and
averaging that away would hide it).

Also runs SHAP on XGBoost (global + a per-ticker "latest prediction"
explanation) and writes the artifacts the Flask backend needs: one
prediction and one history series PER TICKER, so the frontend can offer
a ticker selector.

CRITICAL: LEAKAGE GUARD -- see 05_train_model.py's docstring. Feature
columns come from models/metadata.json (written by 05), never
reconstructed by hand here.

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
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "volatility"
MODELS_DIR = PROJECT_ROOT / "models" / "volatility"
BACKEND_DATA_DIR = PROJECT_ROOT / "backend" / "data" / "volatility"


def load_artifacts():
    test_path = PROCESSED_DATA_DIR / "test.csv"
    meta_path = MODELS_DIR / "metadata.json"

    if not test_path.exists():
        print("ERROR: test.csv not found. Run 04_train_test_split.py first.",
              file=sys.stderr)
        sys.exit(1)
    if not meta_path.exists():
        print("ERROR: models/metadata.json not found. Run 05_train_model.py first.",
              file=sys.stderr)
        sys.exit(1)

    test = pd.read_csv(test_path, index_col="Date", parse_dates=True)
    with open(meta_path) as f:
        metadata = json.load(f)

    log_reg = joblib.load(MODELS_DIR / "logistic_regression.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")

    return test, metadata, log_reg, scaler, xgb


def build_feature_matrix(df: pd.DataFrame, numeric_cols: list, ticker_dummy_cols: list):
    numeric = df[numeric_cols]
    dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    dummies = dummies.reindex(columns=ticker_dummy_cols, fill_value=0)
    return pd.concat([numeric, dummies], axis=1)


def evaluate_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
    }


def print_metrics(name: str, metrics: dict):
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")


def compute_shap_explanations(xgb_model, X_test: pd.DataFrame, feature_cols: list):
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    global_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)
    return explainer, shap_values, global_importance


def top_features_for_row(shap_row: np.ndarray, feature_cols: list, n: int = 5) -> list:
    ranked = sorted(
        zip(feature_cols, shap_row), key=lambda x: -abs(x[1])
    )[:n]
    return [
        {"feature": feat, "shap_value": round(float(val), 4)}
        for feat, val in ranked
        if not feat.startswith("ticker_")  # ticker dummies aren't a
        # meaningful "reason" to show a PM -- they encode WHICH ticker
        # this is, not a market condition, so they're excluded from
        # the plain-language explanation surfaced to the frontend.
    ]


def main():
    test, metadata, log_reg, scaler, xgb = load_artifacts()
    numeric_cols = metadata["numeric_feature_cols"]
    ticker_dummy_cols = metadata["ticker_dummy_cols"]
    feature_cols = metadata["feature_cols"]
    basket = metadata["basket"]

    X_test = build_feature_matrix(test, numeric_cols, ticker_dummy_cols)
    y_test = test["label"]

    # --- Logistic Regression, pooled + per-ticker ---
    X_test_scaled = scaler.transform(X_test)
    lr_pred = log_reg.predict(X_test_scaled)
    lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    lr_pooled = evaluate_metrics(y_test, lr_pred, lr_proba)
    print_metrics("Logistic Regression -- TEST (pooled)", lr_pooled)

    # --- XGBoost, pooled + per-ticker ---
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_pooled = evaluate_metrics(y_test, xgb_pred, xgb_proba)
    print_metrics("XGBoost -- TEST (pooled)", xgb_pooled)

    print("\nXGBoost -- TEST, per ticker:")
    xgb_per_ticker = {}
    for ticker in basket:
        mask = (test["ticker"] == ticker).values
        if mask.sum() == 0:
            continue
        m = evaluate_metrics(y_test[mask], xgb_pred[mask], xgb_proba[mask])
        xgb_per_ticker[ticker] = m
        print(f"  {ticker:>4}: accuracy={m['accuracy']:.3f}  "
              f"roc_auc={m['roc_auc']:.3f}  recall={m['recall']:.3f}")

    # --- SHAP (XGBoost) ---
    explainer, shap_values, global_importance = compute_shap_explanations(
        xgb, X_test, feature_cols
    )
    print("\nSHAP global feature importance (mean |SHAP value|, pooled):")
    for feat, val in global_importance.items():
        print(f"  {feat:>18}: {val:.4f}")

    production_model_name = (
        "xgboost" if xgb_pooled["roc_auc"] >= lr_pooled["roc_auc"] else "logistic_regression"
    )
    print(f"\nProduction model selected: {production_model_name}")

    # --- Write per-ticker artifacts for the Flask backend ---
    BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    latest_predictions = {}
    histories = {}
    for ticker in basket:
        # Locate this ticker's rows via a boolean mask on the
        # already positionally-aligned X_test/xgb_pred/xgb_proba/
        # shap_values arrays -- verified correct against test.iloc[]
        # during development (see project notes).
        mask = (test["ticker"] == ticker).values
        if mask.sum() == 0:
            continue
        ticker_positions = np.where(mask)[0]

        last_pos = ticker_positions[-1]  # most recent row for this ticker
        last_date = test.index[last_pos]

        latest_predictions[ticker] = {
            "date": last_date.strftime("%Y-%m-%d"),
            "regime": "high" if xgb_pred[last_pos] == 1 else "low",
            "probability": round(float(xgb_proba[last_pos]), 4),
            "top_features": top_features_for_row(shap_values[last_pos], feature_cols),
            "model": "xgboost",
            "note": None,
        }

        histories[ticker] = {
            "dates": [d.strftime("%Y-%m-%d") for d in test.index[ticker_positions]],
            "prices": [round(float(p), 2) for p in test["Close"].values[ticker_positions]],
            "regimes": ["high" if xgb_pred[p] == 1 else "low" for p in ticker_positions],
        }

    with open(BACKEND_DATA_DIR / "latest_predictions.json", "w") as f:
        json.dump(latest_predictions, f, indent=2)
    with open(BACKEND_DATA_DIR / "histories.json", "w") as f:
        json.dump(histories, f, indent=2)

    final_metadata = {
        **metadata,
        "test_metrics_pooled": {
            "logistic_regression": lr_pooled,
            "xgboost": xgb_pooled,
        },
        "test_metrics_by_ticker": xgb_per_ticker,
        "shap_global_importance": global_importance.round(4).to_dict(),
        "production_model": production_model_name,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(final_metadata, f, indent=2)

    print(f"\nSaved backend artifacts (per ticker: {', '.join(basket)}) -> "
          f"{BACKEND_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
