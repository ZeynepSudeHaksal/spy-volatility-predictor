"""
06_evaluate.py

Final, honest evaluation on the test split -- the one chunk of data that
has not influenced any decision so far (not used in training, not used
in picking between Logistic Regression and XGBoost, not eyeballed
during feature engineering). This is the number that actually means
something.

Also:
- Runs SHAP on the XGBoost model for explainability: both global
  ("which features matter most overall") and local ("why did the model
  make THIS specific prediction for THIS specific day") explanations.
  SHAP, not raw feature importance, is the standard for the local
  explanation, because it decomposes a single prediction into
  per-feature contributions that sum to that prediction -- exactly the
  "why" a PM/analyst-facing tool needs to show.
- Picks a "production" model (whichever has the higher test ROC-AUC)
  and writes out the artifacts the Flask backend needs to stop
  returning placeholder nulls: the latest prediction (with its SHAP
  explanation) and a history series for the frontend's chart.

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
BACKEND_DATA_DIR = PROJECT_ROOT / "backend" / "data"


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


def evaluate_on_test(name: str, y_true, y_pred, y_proba) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    print(f"\n{name} -- TEST SET (held out, never touched until now):")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    print("\n  Confusion matrix (rows=actual, cols=predicted, [0,1]x[0,1]):")
    print(f"  {confusion_matrix(y_true, y_pred)}")
    print("\n  Full classification report:")
    print(classification_report(y_true, y_pred, target_names=["low-vol", "high-vol"]))
    return metrics


def compare_val_vs_test(val_metrics: dict, test_metrics: dict, model_name: str):
    """
    A meaningful gap between val and test ROC-AUC is a warning sign --
    it suggests the model (or our feature/label choices) is overfit to
    the validation period's specific regime rather than learning
    something that generalizes.
    """
    val_auc = val_metrics["roc_auc"]
    test_auc = test_metrics["roc_auc"]
    gap = val_auc - test_auc
    print(f"\n{model_name}: val ROC-AUC {val_auc:.4f} -> test ROC-AUC {test_auc:.4f} "
          f"(gap: {gap:+.4f})")
    if abs(gap) > 0.10:
        print(
            "  Note: that's a fairly large val/test gap. Worth treating "
            "test performance as the more trustworthy number, and being "
            "cautious about reading too much into val-set tuning."
        )


def compute_shap_explanations(xgb_model, X_test: pd.DataFrame, feature_cols: list):
    """Global (mean |SHAP|) and local (most recent row) explanations."""
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    global_importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=feature_cols
    ).sort_values(ascending=False)

    print("\nSHAP global feature importance (mean |SHAP value| across test set):")
    for feat, val in global_importance.items():
        print(f"  {feat:>18}: {val:.4f}")

    latest_shap = shap_values[-1]
    latest_top_features = sorted(
        [
            {"feature": feat, "shap_value": round(float(val), 4)}
            for feat, val in zip(feature_cols, latest_shap)
        ],
        key=lambda x: -abs(x["shap_value"]),
    )[:5]

    return global_importance, latest_top_features


def main():
    test, metadata, log_reg, scaler, xgb = load_artifacts()
    feature_cols = metadata["feature_cols"]
    X_test, y_test = test[feature_cols], test["label"]

    # --- Logistic Regression on test ---
    X_test_scaled = scaler.transform(X_test)
    lr_pred = log_reg.predict(X_test_scaled)
    lr_proba = log_reg.predict_proba(X_test_scaled)[:, 1]
    lr_test_metrics = evaluate_on_test("Logistic Regression", y_test, lr_pred, lr_proba)
    compare_val_vs_test(
        metadata["val_metrics"]["logistic_regression"], lr_test_metrics,
        "Logistic Regression",
    )

    # --- XGBoost on test ---
    xgb_pred = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_test_metrics = evaluate_on_test("XGBoost", y_test, xgb_pred, xgb_proba)
    compare_val_vs_test(
        metadata["val_metrics"]["xgboost"], xgb_test_metrics, "XGBoost"
    )

    # --- SHAP explainability (XGBoost) ---
    global_importance, latest_top_features = compute_shap_explanations(
        xgb, X_test, feature_cols
    )

    # --- Pick the production model by test ROC-AUC ---
    production_model_name = (
        "xgboost" if xgb_test_metrics["roc_auc"] >= lr_test_metrics["roc_auc"]
        else "logistic_regression"
    )
    print(f"\nProduction model selected: {production_model_name} "
          f"(higher test ROC-AUC)")

    # --- Write artifacts for the Flask backend ---
    BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)

    latest_date = test.index[-1]
    latest_regime = "high" if xgb_pred[-1] == 1 else "low"
    latest_prediction = {
        "date": latest_date.strftime("%Y-%m-%d"),
        "regime": latest_regime,
        "probability": round(float(xgb_proba[-1]), 4),
        "top_features": latest_top_features,
        "model": "xgboost",
        "note": None,
    }
    with open(BACKEND_DATA_DIR / "latest_prediction.json", "w") as f:
        json.dump(latest_prediction, f, indent=2)

    history = {
        "dates": [d.strftime("%Y-%m-%d") for d in test.index],
        "prices": [round(float(p), 2) for p in test["Close"]],
        "regimes": ["high" if p == 1 else "low" for p in xgb_pred],
    }
    with open(BACKEND_DATA_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_metadata = {
        **metadata,
        "test_metrics": {
            "logistic_regression": lr_test_metrics,
            "xgboost": xgb_test_metrics,
        },
        "shap_global_importance": global_importance.round(4).to_dict(),
        "production_model": production_model_name,
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(final_metadata, f, indent=2)

    print(f"\nSaved backend artifacts -> {BACKEND_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print("Updated models/metadata.json with test metrics + SHAP importance + "
          "production model choice.")
    print("\nDone. Next: wire backend/app.py to read these JSON files instead of "
          "returning placeholder nulls.")


if __name__ == "__main__":
    main()
