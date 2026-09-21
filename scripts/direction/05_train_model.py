"""
05_train_model.py

Trains Logistic Regression and XGBoost on the walk-forward train split
(pooled across the basket, ticker as a one-hot feature -- same design
as the volatility pipeline (scripts/volatility/), for the same reasons: shared
learning across tickers, ticker-specific effects still captured via the
dummy columns, one artifact instead of five).

THE BASELINE THAT ACTUALLY MATTERS HERE
--------------------------------------------
For volatility regime, a naive majority-class baseline was a fair
comparison. For DIRECTION, "always predict up" is the baseline that
actually matters, because it exploits real, persistent positive drift
in equity indices -- and as 03_label_data.py's up-rate printout showed,
that baseline often sits in the high 50s%, not 50%. This script prints
it explicitly, separately from the generic DummyClassifier baseline
(which happens to reduce to the same thing when "up" is the majority
class in train, but is computed and shown separately for clarity).
A model's accuracy has to clear the ALWAYS-UP bar, not 50%, before
it demonstrates any real skill -- and even then, ROC-AUC and the
val/test-split recall on each class matter more than accuracy for
judging whether that's a stable edge or an artifact of one drift-y
period.

CRITICAL: LEAKAGE GUARD -- forward_return_5d is the value the label
was built from and must never be a model input. NUMERIC_FEATURE_COLS
is an explicit allow-list for exactly this reason.

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
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "direction"
MODELS_DIR = PROJECT_ROOT / "models" / "direction"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF"]

NUMERIC_FEATURE_COLS = [
    "momentum_5d", "momentum_10d", "momentum_20d",
    "rsi_14", "macd", "macd_hist",
    "price_vs_ma20", "price_vs_ma50",
    "volume_ratio", "realized_vol_20d",
    "vix_level", "vix_spy_spread", "log_return",
]
LABEL_COL = "label"


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / f"{name}.csv"
    if not path.exists():
        print(f"ERROR: {name}.csv not found. Run 04_train_test_split.py first.",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    missing = [c for c in NUMERIC_FEATURE_COLS + [LABEL_COL, "ticker"] if c not in df.columns]
    if missing:
        print(f"ERROR: {name}.csv missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    return df


def build_feature_matrix(df: pd.DataFrame, ticker_dummy_cols=None):
    numeric = df[NUMERIC_FEATURE_COLS]
    dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    if ticker_dummy_cols is not None:
        dummies = dummies.reindex(columns=ticker_dummy_cols, fill_value=0)
        cols_out = ticker_dummy_cols
    else:
        cols_out = list(dummies.columns)
    X = pd.concat([numeric, dummies], axis=1)
    return X, cols_out


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
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

    X_train, ticker_dummy_cols = build_feature_matrix(train)
    y_train = train[LABEL_COL]
    X_val, _ = build_feature_matrix(val, ticker_dummy_cols=ticker_dummy_cols)
    y_val = val[LABEL_COL]

    all_feature_cols = NUMERIC_FEATURE_COLS + ticker_dummy_cols

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    # --- The baseline that matters: always predict "up" ---
    always_up_pred = np.ones(len(y_val), dtype=int)
    always_up_acc = accuracy_score(y_val, always_up_pred)
    print(f"\n'Always predict up' baseline accuracy on val: {always_up_acc:.4f}")
    print("(this reflects real positive drift in the basket -- a real model "
          "must beat THIS, not 50%)")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_val, dummy.predict(X_val))
    print(f"Majority-class DummyClassifier baseline accuracy on val: {dummy_acc:.4f}")

    # --- Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    lr_pred = log_reg.predict(X_val_scaled)
    lr_proba = log_reg.predict_proba(X_val_scaled)[:, 1]
    lr_metrics = evaluate("Logistic Regression (val, pooled)", y_val, lr_pred, lr_proba)

    print("\n  Coefficients (standardized):")
    for feat, coef in sorted(zip(all_feature_cols, log_reg.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"    {feat:>18}: {coef:+.4f}")

    # --- XGBoost ---
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_val)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_metrics = evaluate("XGBoost (val, pooled)", y_val, xgb_pred, xgb_proba)

    print("\n  Feature importances:")
    for feat, imp in sorted(zip(all_feature_cols, xgb.feature_importances_), key=lambda x: -x[1]):
        print(f"    {feat:>18}: {imp:.4f}")

    print(f"\nSummary vs. baselines (val accuracy):")
    print(f"  Always-up baseline:  {always_up_acc:.4f}")
    print(f"  Logistic Regression: {lr_metrics['accuracy']:.4f}  "
          f"({'beats' if lr_metrics['accuracy'] > always_up_acc else 'does NOT beat'} baseline)")
    print(f"  XGBoost:             {xgb_metrics['accuracy']:.4f}  "
          f"({'beats' if xgb_metrics['accuracy'] > always_up_acc else 'does NOT beat'} baseline)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(log_reg, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")

    metadata = {
        "basket": BASKET,
        "numeric_feature_cols": NUMERIC_FEATURE_COLS,
        "ticker_dummy_cols": ticker_dummy_cols,
        "feature_cols": all_feature_cols,
        "label_col": LABEL_COL,
        "always_up_baseline_val_accuracy": always_up_acc,
        "val_metrics": {"logistic_regression": lr_metrics, "xgboost": xgb_metrics},
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models + metadata -> {MODELS_DIR.relative_to(PROJECT_ROOT)}")
    print("\nDone. Next: run 06_evaluate.py")


if __name__ == "__main__":
    main()
