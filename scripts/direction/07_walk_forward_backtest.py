"""
07_walk_forward_backtest.py

Replaces the single static train/val/test evaluation with a proper
ROLLING walk-forward backtest: instead of training once on old data and
testing once on new data, this retrains fresh on a trailing window
every few months and walks that forward through the full history,
collecting genuinely out-of-sample predictions the whole way.

WHY THIS EXISTS
--------------------
A single train/test split (scripts 04-06) tests one thing: "does a
model trained on 2001-2023 work on 2023-2026?" If the answer is no,
that could mean the approach doesn't work at all -- OR it could mean
the market regime shifted and a model frozen on old data just can't
keep up, which is a solvable problem (retrain periodically), not a
dead end. This script tests the SECOND, more realistic scenario: "if
I'd been retraining this model every 6 months on the trailing 5 years,
using it, then retraining again, how would it actually have done?"

Two things come out of this:
1. A much more statistically robust accuracy estimate (dozens of folds
   instead of one lucky-or-unlucky split) -- and per-fold results over
   time, so a regime-shift pattern (some periods harder than others)
   shows up directly in the data instead of being guessed at.
2. The actual "production" model: trained on the MOST RECENT trailing
   window, which is what gets deployed to the Flask backend --
   replacing the single frozen model 05/06 would have produced.

scripts/04_train_test_split.py through 06_evaluate.py still exist and
still work, for quick single-split iteration during development. This
script is the rigorous version for judging whether the approach
actually works, and for producing the model that actually ships.

Usage:
    python scripts/direction/07_walk_forward_backtest.py
    python scripts/direction/07_walk_forward_backtest.py --window-years 5 --step-months 6
    python scripts/direction/07_walk_forward_backtest.py --exclude-features vol_regime_probability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for cv_utils
from cv_utils import rolling_walk_forward_splits

from common import (
    BASKET, LABEL_COL, MODELS_DIR, NUMERIC_FEATURE_COLS,
    PROCESSED_DATA_DIR, PROJECT_ROOT, build_feature_matrix,
)

BACKEND_DATA_DIR = PROJECT_ROOT / "backend" / "data" / "direction"
FORWARD_WINDOW = 5  # must match 03_label_data.py


def load_labeled_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "labeled_data.csv"
    if not path.exists():
        print("ERROR: labeled_data.csv not found. Run 03_label_data.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def get_default_hyperparams():
    path = MODELS_DIR / "tuned_hyperparameters.json"
    if not path.exists():
        return {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05}, {"C": 1.0}
    with open(path) as f:
        tuned = json.load(f)
    return tuned["xgboost"]["params"], tuned["logistic_regression"]["params"]


def train_and_predict_fold(train_df, test_df, numeric_cols, ticker_dummy_cols,
                             xgb_params, lr_params):
    """Trains fresh LR and XGBoost on train_df, predicts on test_df.
    Returns (xgb_proba, lr_proba, y_test) as arrays aligned to test_df's
    row order."""
    X_train, _ = build_feature_matrix(train_df, ticker_dummy_cols=ticker_dummy_cols, numeric_cols=numeric_cols)
    X_test, _ = build_feature_matrix(test_df, ticker_dummy_cols=ticker_dummy_cols, numeric_cols=numeric_cols)
    y_train = train_df[LABEL_COL]
    y_test = test_df[LABEL_COL]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
        **xgb_params,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]

    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, **lr_params)
    lr.fit(scaler.transform(X_train), y_train)
    lr_proba = lr.predict_proba(scaler.transform(X_test))[:, 1]

    return xgb_proba, lr_proba, y_test.values


def main():
    parser = argparse.ArgumentParser(description="Rolling walk-forward backtest.")
    parser.add_argument("--window-years", type=float, default=5.0,
                         help="Trailing training window size, in years.")
    parser.add_argument("--step-months", type=float, default=6.0,
                         help="How far each fold steps forward.")
    parser.add_argument("--exclude-features", type=str, default="",
                         help="Comma-separated feature names to drop, e.g. for ablation.")
    args = parser.parse_args()

    excluded = [f.strip() for f in args.exclude_features.split(",") if f.strip()]
    numeric_cols = [c for c in NUMERIC_FEATURE_COLS if c not in excluded]
    if excluded:
        print(f"Excluding {excluded} from all folds.")

    df = load_labeled_data()
    xgb_params, lr_params = get_default_hyperparams()

    ticker_dummy_cols = [f"ticker_{t}" for t in BASKET]

    splits = rolling_walk_forward_splits(
        dates=df.reset_index()["Date"],
        window_years=args.window_years,
        step_months=args.step_months,
        purge_gap=FORWARD_WINDOW,
        ticker=df.reset_index()["ticker"],
    )
    print(f"Built {len(splits)} rolling folds "
          f"({args.window_years}y window, {args.step_months}mo step, "
          f"{FORWARD_WINDOW}-day purge gap).\n")

    df_reset = df.reset_index()

    all_xgb_proba, all_lr_proba, all_y_true = [], [], []
    all_dates, all_tickers, all_close = [], [], []
    fold_records = []

    for i, (train_idx, test_idx, test_start, test_end) in enumerate(splits):
        train_fold = df_reset.iloc[train_idx]
        test_fold = df_reset.iloc[test_idx]

        xgb_proba, lr_proba, y_test = train_and_predict_fold(
            train_fold, test_fold, numeric_cols, ticker_dummy_cols, xgb_params, lr_params
        )

        fold_always_up = float(y_test.mean())
        fold_xgb_pred = (xgb_proba > 0.5).astype(int)
        fold_acc = accuracy_score(y_test, fold_xgb_pred)
        fold_auc = roc_auc_score(y_test, xgb_proba) if len(set(y_test)) > 1 else float("nan")

        fold_records.append({
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
            "n_test": len(test_idx),
            "always_up_baseline": round(fold_always_up, 4),
            "xgb_accuracy": round(fold_acc, 4),
            "xgb_roc_auc": round(fold_auc, 4) if not np.isnan(fold_auc) else None,
        })
        print(f"Fold {i:>2} [{test_start.date()} to {test_end.date()}): "
              f"n={len(test_idx):>4}  always_up={fold_always_up:.3f}  "
              f"xgb_acc={fold_acc:.3f}  xgb_auc={fold_auc:.3f}")

        all_xgb_proba.append(xgb_proba)
        all_lr_proba.append(lr_proba)
        all_y_true.append(y_test)
        all_dates.append(test_fold["Date"].values)
        all_tickers.append(test_fold["ticker"].values)
        all_close.append(test_fold["Close"].values)

    all_xgb_proba = np.concatenate(all_xgb_proba)
    all_lr_proba = np.concatenate(all_lr_proba)
    all_y_true = np.concatenate(all_y_true)
    all_dates = np.concatenate(all_dates)
    all_tickers = np.concatenate(all_tickers)
    all_close = np.concatenate(all_close)

    overall_always_up = all_y_true.mean()
    xgb_pred = (all_xgb_proba > 0.5).astype(int)
    lr_pred = (all_lr_proba > 0.5).astype(int)

    print(f"\n{'='*60}")
    print(f"AGGREGATE WALK-FORWARD RESULTS (all {len(all_y_true):,} out-of-fold predictions)")
    print(f"{'='*60}")
    print(f"Always-up baseline: {overall_always_up:.4f}")
    print(f"XGBoost: accuracy={accuracy_score(all_y_true, xgb_pred):.4f}  "
          f"roc_auc={roc_auc_score(all_y_true, all_xgb_proba):.4f}")
    print(f"Logistic Regression: accuracy={accuracy_score(all_y_true, lr_pred):.4f}  "
          f"roc_auc={roc_auc_score(all_y_true, all_lr_proba):.4f}")

    print(f"\nPer-ticker aggregate (XGBoost):")
    for ticker in BASKET:
        mask = all_tickers == ticker
        if mask.sum() == 0:
            continue
        t_acc = accuracy_score(all_y_true[mask], xgb_pred[mask])
        t_auc = roc_auc_score(all_y_true[mask], all_xgb_proba[mask]) if len(set(all_y_true[mask])) > 1 else float("nan")
        t_baseline = all_y_true[mask].mean()
        print(f"  {ticker:>4}: n={mask.sum():>5}  always_up={t_baseline:.3f}  "
              f"acc={t_acc:.3f}  auc={t_auc:.3f}")

    # --- Train the FINAL production model on the most recent trailing window ---
    print(f"\nTraining production model on the most recent {args.window_years}-year window...")
    last_date = df.index.max()
    window = pd.Timedelta(days=int(args.window_years * 365.25))
    prod_train = df_reset[df_reset["Date"] > (last_date - window)]

    X_prod, prod_ticker_cols = build_feature_matrix(prod_train, ticker_dummy_cols=None, numeric_cols=numeric_cols)
    y_prod = prod_train[LABEL_COL]

    n_pos = y_prod.sum()
    n_neg = len(y_prod) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    prod_xgb = XGBClassifier(
        scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42, **xgb_params,
    )
    prod_xgb.fit(X_prod, y_prod)

    prod_scaler = StandardScaler().fit(X_prod)
    prod_lr = LogisticRegression(class_weight="balanced", max_iter=1000, **lr_params)
    prod_lr.fit(prod_scaler.transform(X_prod), y_prod)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(prod_xgb, MODELS_DIR / "xgboost.joblib")
    joblib.dump(prod_lr, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(prod_scaler, MODELS_DIR / "scaler.joblib")

    metadata = {
        "basket": BASKET,
        "numeric_feature_cols": numeric_cols,
        "ticker_dummy_cols": prod_ticker_cols,
        "feature_cols": numeric_cols + prod_ticker_cols,
        "label_col": LABEL_COL,
        "xgb_params_used": xgb_params,
        "lr_params_used": lr_params,
        "walk_forward": {
            "window_years": args.window_years,
            "step_months": args.step_months,
            "n_folds": len(splits),
            "always_up_baseline": round(float(overall_always_up), 4),
            "xgb_accuracy": round(float(accuracy_score(all_y_true, xgb_pred)), 4),
            "xgb_roc_auc": round(float(roc_auc_score(all_y_true, all_xgb_proba)), 4),
            "lr_accuracy": round(float(accuracy_score(all_y_true, lr_pred)), 4),
            "lr_roc_auc": round(float(roc_auc_score(all_y_true, all_lr_proba)), 4),
            "fold_records": fold_records,
        },
        "production_model": "xgboost" if roc_auc_score(all_y_true, all_xgb_proba) >= roc_auc_score(all_y_true, all_lr_proba) else "logistic_regression",
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --- Write backend artifacts ---
    # History: every out-of-fold walk-forward prediction, genuinely
    # out-of-sample the whole way -- a much more honest history than a
    # single train/test split could offer.
    BACKEND_DATA_DIR.mkdir(parents=True, exist_ok=True)
    histories = {}
    for ticker in BASKET:
        mask = all_tickers == ticker
        if mask.sum() == 0:
            continue
        order = np.argsort(all_dates[mask])
        histories[ticker] = {
            "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in all_dates[mask][order]],
            "prices": [round(float(p), 2) for p in all_close[mask][order]],
            "directions": ["up" if p == 1 else "down" for p in xgb_pred[mask][order]],
        }
    with open(BACKEND_DATA_DIR / "histories.json", "w") as f:
        json.dump(histories, f, indent=2)

    # Latest prediction: score the production model on the single most
    # recent row of FEATURES (not labeled_data.csv, which necessarily
    # excludes the last few days -- they don't have a completed
    # forward-return window yet, which is exactly what we're predicting).
    features_path = PROCESSED_DATA_DIR / "features.csv"
    features = pd.read_csv(features_path, index_col="Date", parse_dates=True)

    import shap
    explainer = shap.TreeExplainer(prod_xgb)

    latest_predictions = {}
    for ticker in BASKET:
        ticker_features = features[features["ticker"] == ticker].sort_index()
        if len(ticker_features) == 0:
            continue
        latest_row = ticker_features.iloc[[-1]]
        X_latest, _ = build_feature_matrix(latest_row, ticker_dummy_cols=prod_ticker_cols, numeric_cols=numeric_cols)
        proba = prod_xgb.predict_proba(X_latest)[0, 1]
        shap_values = explainer.shap_values(X_latest)[0]
        top_features = sorted(
            zip(numeric_cols + prod_ticker_cols, shap_values), key=lambda x: -abs(x[1])
        )[:5]
        # proba is P(up) from the model. "probability" here means
        # "confidence in the predicted direction" -- when the call is
        # "down", that's P(down) = 1 - proba, not the raw P(up). Without
        # this flip, a confident "down" call (e.g. P(up)=0.18) rendered
        # on the dashboard as "Down expected, 18% probability", which
        # reads as 18% confidence in a call the model is actually 82%
        # confident in.
        direction = "up" if proba > 0.5 else "down"
        confidence = proba if direction == "up" else 1 - proba
        latest_predictions[ticker] = {
            "date": latest_row.index[0].strftime("%Y-%m-%d"),
            "direction": direction,
            "probability": round(float(confidence), 4),
            "always_up_baseline": round(float(overall_always_up), 4),
            "top_features": [
                {"feature": f, "shap_value": round(float(v), 4)}
                for f, v in top_features if not f.startswith("ticker_")
            ],
            "model": "xgboost (walk-forward production model)",
            "note": None,
        }
    with open(BACKEND_DATA_DIR / "latest_predictions.json", "w") as f:
        json.dump(latest_predictions, f, indent=2)

    print(f"\nSaved production model + metadata -> {MODELS_DIR}")
    print(f"Saved backend artifacts (walk-forward history + latest predictions) -> {BACKEND_DATA_DIR}")
    print("\nDone.")


if __name__ == "__main__":
    main()
