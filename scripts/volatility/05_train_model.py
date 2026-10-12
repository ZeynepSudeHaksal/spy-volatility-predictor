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
about general volatility clustering while still picking up
ticker-specific effects through the ticker dummy variables.

HYPERPARAMETERS: TUNED IF AVAILABLE, SENSIBLE DEFAULTS OTHERWISE
------------------------------------------------------------------------
If scripts/volatility/05a_tune_hyperparameters.py has been run, this
script loads its output (models/volatility/tuned_hyperparameters.json)
and uses those hyperparameters. Otherwise it falls back to the
hand-picked defaults below. Either way, VAL remains untouched by
hyperparameter selection -- tuning (05a) searches only within TRAIN via
purged walk-forward CV; val here is purely for evaluating whichever
hyperparameters were ultimately used, exactly as before.

A NOTE ON WHAT "GOOD" LOOKS LIKE HERE
----------------------------------------
ROC-AUC is still the primary metric -- it's threshold-independent and
measures whether the model's probabilities actually rank high-vol days
above normal ones, which is what matters for a risk-flagging tool.

Raw accuracy at the default 0.5 cut is misleading here: both models use
class weighting (LogisticRegression(class_weight="balanced") / XGBoost's
scale_pos_weight) to deliberately favor recall on the minority high-vol
class, which pushes predicted probabilities up and can make accuracy
come in BELOW the naive "always predict the majority class" baseline --
even when ROC-AUC clearly shows real signal. select_f1_threshold()
(common.py) picks a better cut on VAL and 06_evaluate.py applies it to
TEST, so the accuracy numbers reported are an honest, apples-to-apples
comparison against that baseline rather than an artifact of the 0.5
default. See 06_evaluate.py for the per-ticker breakdown.

Usage:
    python scripts/volatility/05_train_model.py
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import (
    BASKET, LABEL_COL, MODELS_DIR, NUMERIC_FEATURE_COLS,
    build_feature_matrix, load_split, select_f1_threshold,
)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
}
DEFAULT_LR_PARAMS = {"C": 1.0}


def load_tuned_params():
    """
    Returns (xgb_params, lr_params). Falls back to the hand-picked
    defaults above if 05a_tune_hyperparameters.py hasn't been run yet
    -- this script always works, tuned or not.
    """
    path = MODELS_DIR / "tuned_hyperparameters.json"
    if not path.exists():
        print("No tuned_hyperparameters.json found -- using default "
              "hyperparameters. Run 05a_tune_hyperparameters.py first "
              "for tuned ones.")
        return dict(DEFAULT_XGB_PARAMS), dict(DEFAULT_LR_PARAMS)

    with open(path) as f:
        tuned = json.load(f)
    print(f"Loaded tuned hyperparameters from {path.name} "
          f"(CV ROC-AUC: xgb={tuned['xgboost']['cv_roc_auc']:.4f}, "
          f"lr={tuned['logistic_regression']['cv_roc_auc']:.4f})")
    return tuned["xgboost"]["params"], tuned["logistic_regression"]["params"]


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
    print(f"Train class balance -- high-vol: {y_train.mean() * 100:.1f}%")

    xgb_params, lr_params = load_tuned_params()

    # --- Naive baseline ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)
    naive_baseline_accuracy = accuracy_score(y_val, dummy_pred)
    print(f"\nNaive baseline (always predict majority class):")
    print(f"  accuracy: {naive_baseline_accuracy:.4f}")

    # --- Logistic Regression (scaled, class-weighted) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, **lr_params)
    log_reg.fit(X_train_scaled, y_train)
    lr_proba = log_reg.predict_proba(X_val_scaled)[:, 1]

    # class_weight="balanced" shifts LR's probabilities toward the
    # minority (high-vol) class, which is good for recall but can tank
    # accuracy at the default 0.5 cut -- see select_f1_threshold's
    # docstring. Pick the threshold that actually maximizes accuracy on
    # VAL (never touches test), and use it everywhere downstream,
    # including 06_evaluate.py.
    lr_threshold = select_f1_threshold(y_val, lr_proba)
    lr_pred = (lr_proba >= lr_threshold).astype(int)
    lr_metrics = evaluate("Logistic Regression (val, pooled)", y_val, lr_pred, lr_proba)
    print(f"  (decision threshold selected on val: {lr_threshold:.2f}, "
          f"vs. default 0.5)")

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
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        **xgb_params,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]

    xgb_threshold = select_f1_threshold(y_val, xgb_proba)
    xgb_pred = (xgb_proba >= xgb_threshold).astype(int)
    xgb_metrics = evaluate("XGBoost (val, pooled)", y_val, xgb_pred, xgb_proba)
    print(f"  (decision threshold selected on val: {xgb_threshold:.2f}, "
          f"vs. default 0.5)")

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
        "xgb_params_used": xgb_params,
        "lr_params_used": lr_params,
        "val_metrics": {
            "logistic_regression": lr_metrics,
            "xgboost": xgb_metrics,
        },
        "val_naive_baseline_accuracy": naive_baseline_accuracy,
        # Selected on VAL only (see select_f1_threshold in common.py);
        # 06_evaluate.py reads these and applies them as-is to TEST, so
        # test stays untouched by the selection itself.
        "decision_threshold": {
            "logistic_regression": lr_threshold,
            "xgboost": xgb_threshold,
        },
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models + metadata -> {MODELS_DIR}")
    print("\nDone. Next: run 06_evaluate.py for full test-set evaluation + SHAP.")


if __name__ == "__main__":
    main()
