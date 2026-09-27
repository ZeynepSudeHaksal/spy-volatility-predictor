"""
05_train_model.py

Trains Logistic Regression and XGBoost on the walk-forward train split
(pooled across the basket, ticker as a one-hot feature). See
scripts/volatility/05_train_model.py for the shared design rationale.

HYPERPARAMETERS: TUNED IF AVAILABLE, SENSIBLE DEFAULTS OTHERWISE
------------------------------------------------------------------------
If scripts/direction/05a_tune_hyperparameters.py has been run, this
loads its output (models/direction/tuned_hyperparameters.json) and
uses those hyperparameters; otherwise falls back to hand-picked
defaults. Val remains untouched by the search either way.

THE BASELINE THAT ACTUALLY MATTERS HERE
--------------------------------------------
"Always predict up" -- reflects real historical equity drift, not a
50% coin flip. A model has to clear THAT bar, not 50%, to demonstrate
real skill.

Usage:
    python scripts/direction/05_train_model.py
"""

from __future__ import annotations

import argparse
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
    build_feature_matrix, load_split,
)

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
}
DEFAULT_LR_PARAMS = {"C": 1.0}


def load_tuned_params():
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
    parser = argparse.ArgumentParser(description="Train direction models.")
    parser.add_argument(
        "--exclude-features", type=str, default="",
        help="Comma-separated feature names to drop before training -- "
             "for ablation studies, e.g. --exclude-features vol_regime_probability "
             "to test whether that feature is actually helping.",
    )
    args = parser.parse_args()
    excluded = [f.strip() for f in args.exclude_features.split(",") if f.strip()]
    numeric_cols = [c for c in NUMERIC_FEATURE_COLS if c not in excluded]
    if excluded:
        print(f"Ablation: excluding {excluded} from training.")

    train = load_split("train")
    val = load_split("val")

    X_train, ticker_dummy_cols = build_feature_matrix(train, ticker_dummy_cols=None, numeric_cols=numeric_cols)
    y_train = train[LABEL_COL]
    X_val, _ = build_feature_matrix(val, ticker_dummy_cols=ticker_dummy_cols, numeric_cols=numeric_cols)
    y_val = val[LABEL_COL]

    all_feature_cols = numeric_cols + ticker_dummy_cols

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    xgb_params, lr_params = load_tuned_params()

    always_up_pred = np.ones(len(y_val), dtype=int)
    always_up_acc = accuracy_score(y_val, always_up_pred)
    print(f"\n'Always predict up' baseline accuracy on val: {always_up_acc:.4f}")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_acc = accuracy_score(y_val, dummy.predict(X_val))
    print(f"Majority-class DummyClassifier baseline accuracy on val: {dummy_acc:.4f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, **lr_params)
    log_reg.fit(X_train_scaled, y_train)
    lr_pred = log_reg.predict(X_val_scaled)
    lr_proba = log_reg.predict_proba(X_val_scaled)[:, 1]
    lr_metrics = evaluate("Logistic Regression (val, pooled)", y_val, lr_pred, lr_proba)

    print("\n  Coefficients (standardized):")
    for feat, coef in sorted(zip(all_feature_cols, log_reg.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"    {feat:>18}: {coef:+.4f}")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
        **xgb_params,
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
        "numeric_feature_cols": numeric_cols,
        "ticker_dummy_cols": ticker_dummy_cols,
        "feature_cols": all_feature_cols,
        "label_col": LABEL_COL,
        "xgb_params_used": xgb_params,
        "lr_params_used": lr_params,
        "always_up_baseline_val_accuracy": always_up_acc,
        "val_metrics": {"logistic_regression": lr_metrics, "xgboost": xgb_metrics},
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved models + metadata -> {MODELS_DIR}")
    print("\nDone. Next: run 06_evaluate.py")


if __name__ == "__main__":
    main()