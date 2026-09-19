"""
05_train_model.py

Trains two classifiers on the walk-forward train split and evaluates
both on the (untouched-until-now) validation split:

1. Logistic Regression -- the explainability baseline. Coefficients are
   directly interpretable (a positive coefficient on vix_spy_spread
   means "higher spread pushes the prediction toward high-vol regime"),
   which matters a lot for a PM/analyst-facing tool where "why did the
   model say this" needs a simple answer.

2. XGBoost -- a stronger, non-linear model. Explainability here comes
   from SHAP values (added in 06_evaluate.py) rather than raw
   coefficients, but gradient-boosted trees typically pick up threshold
   effects and feature interactions that logistic regression can't
   (e.g. "volume_ratio only matters when realized_vol_5d is already
   elevated").

A NOTE ON WHAT "GOOD" LOOKS LIKE HERE
----------------------------------------
Unlike price-direction prediction, this is a more genuinely learnable
problem (volatility clustering is real and well documented), so it's
reasonable to expect the model to meaningfully beat a naive baseline --
this script always prints one for honest comparison. But still: judge
results by ROC-AUC and recall on the high-vol class, not raw accuracy.
A model that just predicts the majority class can look deceptively
decent on accuracy alone, especially if a split's classes are imbalanced
(04_train_test_split.py already flagged this can happen across regimes).

CRITICAL: LEAKAGE GUARD
--------------------------
labeled_data.csv (and therefore train.csv/val.csv) contains
forward_vol_5d and trailing_vol_median_60d -- these were used to BUILD
the label and must never be used as model inputs. FEATURE_COLS below is
an explicit allow-list for exactly this reason: it's safer than
"everything except the label," which would silently break the moment
someone adds a new column upstream.

Usage:
    python scripts/05_train_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Explicit allow-list -- see "CRITICAL: LEAKAGE GUARD" above.
FEATURE_COLS = [
    "log_return", "abs_return",
    "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
    "atr", "atr_pct",
    "volume_ratio",
    "vix_level", "vix_spy_spread",
]
LABEL_COL = "label"


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    if not path.exists():
        print(
            f"ERROR: {name}.csv not found. Run 04_train_test_split.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)

    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        print(f"ERROR: {name}.csv is missing expected columns: {missing}",
              file=sys.stderr)
        sys.exit(1)

    return df


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    print(f"\n{name}:")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    return metrics


def main():
    train = load_split("train")
    val = load_split("val")

    X_train, y_train = train[FEATURE_COLS], train[LABEL_COL]
    X_val, y_val = val[FEATURE_COLS], val[LABEL_COL]

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")
    print(f"Train class balance -- high-vol: {y_train.mean() * 100:.1f}%")

    # --- Naive baseline: always predict the majority class from train ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)
    # DummyClassifier's predict_proba is degenerate (all 0 or 1 for the
    # majority class), so ROC-AUC for it is uninformative -- report
    # accuracy only, which is the whole point: show how easy accuracy
    # is to game by comparison.
    print(f"\nNaive baseline (always predict majority class):")
    print(f"  accuracy: {accuracy_score(y_val, dummy_pred):.4f}")

    # --- Logistic Regression (scaled, class-weighted) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    lr_pred = log_reg.predict(X_val_scaled)
    lr_proba = log_reg.predict_proba(X_val_scaled)[:, 1]
    lr_metrics = evaluate("Logistic Regression (val)", y_val, lr_pred, lr_proba)

    print("\n  Coefficients (standardized -- larger magnitude = stronger effect):")
    for feat, coef in sorted(
        zip(FEATURE_COLS, log_reg.coef_[0]), key=lambda x: -abs(x[1])
    ):
        print(f"    {feat:>18}: {coef:+.4f}")

    # --- XGBoost (unscaled -- tree splits don't need it; class-weighted
    #     via scale_pos_weight, XGBoost's equivalent of class_weight) ---
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_val)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_metrics = evaluate("XGBoost (val)", y_val, xgb_pred, xgb_proba)

    print("\n  Feature importances:")
    for feat, imp in sorted(
        zip(FEATURE_COLS, xgb.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"    {feat:>18}: {imp:.4f}")

    # --- Save everything 06_evaluate.py and the Flask backend will need ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(log_reg, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")

    metadata = {
        "feature_cols": FEATURE_COLS,
        "label_col": LABEL_COL,
        "val_metrics": {
            "logistic_regression": lr_metrics,
            "xgboost": xgb_metrics,
        },
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models + metadata -> {MODELS_DIR.relative_to(PROJECT_ROOT)}")
    print("\nDone. Next: run 06_evaluate.py for full test-set evaluation + SHAP.")


if __name__ == "__main__":
    main()
