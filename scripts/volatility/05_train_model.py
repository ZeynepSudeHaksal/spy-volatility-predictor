"""
05_train_model.py

Trains two classifiers on the walk-forward train split (pooled across
the whole basket) and evaluates both on the validation split:

1. Logistic Regression -- the explainability baseline.
2. XGBoost -- a stronger, non-linear model, explained via SHAP in
   06_evaluate.py.

ONE POOLED MODEL, NOT FIVE SEPARATE ONES
---------------------------------------------
Rather than training a separate model per ticker, this trains a single
model on all five tickers stacked together, with 'ticker' included as
a one-hot encoded feature. This lets the model share what it learns
about general volatility clustering (which shows up similarly across
SPY/QQQ/IWM/DIA/XLF) while still picking up ticker-specific effects
(e.g. IWM running structurally hotter than DIA) through the ticker
dummy variables. It's also a single artifact to maintain rather than
five separately-drifting models.

The ticker dummy columns are fit ONCE on train and that exact column
set is reused (via reindex, filling any ticker missing from a given
split with 0) for val/test and for live prediction in the Flask
backend -- this guarantees the feature matrix always has the same
shape and column order the model was trained on.

A NOTE ON WHAT "GOOD" LOOKS LIKE HERE
----------------------------------------
Judge results by ROC-AUC and recall on the high-vol class, not raw
accuracy -- see 06_evaluate.py for the per-ticker breakdown, since a
pooled model could plausibly do well on the tickers with more/cleaner
signal and mask weaker performance on others if only look at the
pooled number.

CRITICAL: LEAKAGE GUARD
--------------------------
labeled_data.csv contains forward_vol_5d and trailing_vol_median_60d --
used to BUILD the label, never as model inputs. NUMERIC_FEATURE_COLS
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
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "volatility"
MODELS_DIR = PROJECT_ROOT / "models" / "volatility"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF"]

# Explicit allow-list -- see "CRITICAL: LEAKAGE GUARD" above.
NUMERIC_FEATURE_COLS = [
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
        print(f"ERROR: {name}.csv not found. Run 04_train_test_split.py first.",
              file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)

    missing = [c for c in NUMERIC_FEATURE_COLS + [LABEL_COL, "ticker"] if c not in df.columns]
    if missing:
        print(f"ERROR: {name}.csv is missing expected columns: {missing}",
              file=sys.stderr)
        sys.exit(1)
    return df


def build_feature_matrix(df: pd.DataFrame, ticker_dummy_cols: list[str] | None):
    """
    Builds [numeric features] + [one-hot ticker dummies]. If
    ticker_dummy_cols is given (from train), the dummy columns are
    reindexed to exactly match it -- so val/test/live data always
    produces a feature matrix with the same shape and column order the
    model was trained on, even if a ticker is missing from that split.
    """
    numeric = df[NUMERIC_FEATURE_COLS]
    dummies = pd.get_dummies(df["ticker"], prefix="ticker")

    if ticker_dummy_cols is not None:
        dummies = dummies.reindex(columns=ticker_dummy_cols, fill_value=0)
        dummy_cols_out = ticker_dummy_cols
    else:
        dummy_cols_out = list(dummies.columns)

    X = pd.concat([numeric, dummies], axis=1)
    return X, dummy_cols_out


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

    X_train, ticker_dummy_cols = build_feature_matrix(train, ticker_dummy_cols=None)
    y_train = train[LABEL_COL]
    X_val, _ = build_feature_matrix(val, ticker_dummy_cols=ticker_dummy_cols)
    y_val = val[LABEL_COL]

    all_feature_cols = NUMERIC_FEATURE_COLS + ticker_dummy_cols

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")
    print(f"Feature columns ({len(all_feature_cols)}): "
          f"{NUMERIC_FEATURE_COLS} + {ticker_dummy_cols}")
    print(f"Train class balance -- high-vol: {y_train.mean() * 100:.1f}%")

    # --- Naive baseline ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)
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
    lr_metrics = evaluate("Logistic Regression (val, pooled)", y_val, lr_pred, lr_proba)

    print("\n  Coefficients (standardized -- larger magnitude = stronger effect):")
    for feat, coef in sorted(
        zip(all_feature_cols, log_reg.coef_[0]), key=lambda x: -abs(x[1])
    ):
        print(f"    {feat:>18}: {coef:+.4f}")

    # --- XGBoost (unscaled; class-weighted via scale_pos_weight) ---
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
    xgb_metrics = evaluate("XGBoost (val, pooled)", y_val, xgb_pred, xgb_proba)

    print("\n  Feature importances:")
    for feat, imp in sorted(
        zip(all_feature_cols, xgb.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"    {feat:>18}: {imp:.4f}")

    # --- Save everything 06_evaluate.py and the Flask backend will need ---
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
