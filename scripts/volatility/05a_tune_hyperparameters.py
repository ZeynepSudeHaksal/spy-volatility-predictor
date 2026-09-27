"""
05a_tune_hyperparameters.py

Searches for better XGBoost and Logistic Regression hyperparameters
using PURGED WALK-FORWARD CROSS-VALIDATION on the TRAIN split ONLY.
Val and test remain completely untouched by this search -- exactly as
they are in 05_train_model.py -- so nothing here can leak into the
numbers reported later. Val is still there afterward purely to
evaluate whichever hyperparameters this script finds; it never enters
the search itself.

WHY NOT PLAIN K-FOLD CV
----------------------------
Standard k-fold CV shuffles rows into random folds. For this panel of
time series, that's broken for two reasons already established
elsewhere in this pipeline (see 04_train_test_split.py's docstring):
features are autocorrelated across nearby dates, and labels look
FORWARD_WINDOW days into the future, so a naive fold split leaks
information across fold boundaries the same way it would across the
outer train/val/test boundary. The fix -- purged walk-forward CV,
implemented once in scripts/cv_utils.py and imported here -- expands
the training window fold by fold and purges the last few rows of each
ticker's training data before each validation window, so no fold can
see a label that partly reflects its own future.

WHAT GETS SEARCHED
-----------------------
XGBoost: max_depth, n_estimators, learning_rate, min_child_weight,
subsample, colsample_bytree -- the standard levers, with subsample/
colsample included specifically as regularization against overfitting
on financial data. Logistic Regression: C (inverse regularization
strength).

Scoring: ROC-AUC, matching this project's established evaluation
philosophy (see 05_train_model.py, 06_evaluate.py) of not judging by
raw accuracy alone.

Usage:
    python scripts/volatility/05a_tune_hyperparameters.py
    python scripts/volatility/05a_tune_hyperparameters.py --n-splits 5 --n-iter 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for cv_utils
from cv_utils import purged_walk_forward_splits

from common import LABEL_COL, MODELS_DIR, build_feature_matrix, load_split

FORWARD_WINDOW = 5  # must match 03_label_data.py -- the purge gap size

XGB_PARAM_GRID = {
    "clf__max_depth": [3, 4, 5, 6],
    "clf__n_estimators": [100, 200, 300],
    "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "clf__min_child_weight": [1, 3, 5],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.7, 0.85, 1.0],
}
LR_PARAM_GRID = {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}


def main():
    parser = argparse.ArgumentParser(description="Purged walk-forward hyperparameter search.")
    parser.add_argument("--n-splits", type=int, default=4,
                         help="Number of purged walk-forward CV folds within train.")
    parser.add_argument("--n-iter", type=int, default=25,
                         help="Number of random XGBoost hyperparameter combos to try.")
    args = parser.parse_args()

    train = load_split("train")
    X_train, ticker_dummy_cols = build_feature_matrix(train, ticker_dummy_cols=None)
    y_train = train[LABEL_COL]

    cv_splits = purged_walk_forward_splits(
        dates=train.reset_index()["Date"],
        n_splits=args.n_splits,
        purge_gap=FORWARD_WINDOW,
        ticker=train.reset_index()["ticker"],
    )
    print(f"Built {len(cv_splits)} purged walk-forward CV folds within train "
          f"({len(X_train):,} rows).")
    for i, (tr_idx, val_idx) in enumerate(cv_splits):
        print(f"  Fold {i}: train {len(tr_idx):,} rows, val {len(val_idx):,} rows")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # --- XGBoost random search ---
    print(f"\nSearching XGBoost hyperparameters ({args.n_iter} random combos "
          f"x {len(cv_splits)} folds)...")
    xgb_pipeline = Pipeline([
        ("clf", XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
        )),
    ])
    xgb_search = RandomizedSearchCV(
        xgb_pipeline, XGB_PARAM_GRID, n_iter=args.n_iter, cv=cv_splits,
        scoring="roc_auc", random_state=42, n_jobs=-1,
    )
    xgb_search.fit(X_train, y_train)
    xgb_best_params = {k.replace("clf__", ""): v for k, v in xgb_search.best_params_.items()}
    print(f"Best XGBoost params: {xgb_best_params}")
    print(f"Best XGBoost CV ROC-AUC: {xgb_search.best_score_:.4f}")

    # --- Logistic Regression grid search (small grid -- exhaustive is fine) ---
    print(f"\nSearching Logistic Regression hyperparameters "
          f"({len(LR_PARAM_GRID['clf__C'])} values x {len(cv_splits)} folds)...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    lr_search = GridSearchCV(
        lr_pipeline, LR_PARAM_GRID, cv=cv_splits, scoring="roc_auc", n_jobs=-1,
    )
    lr_search.fit(X_train, y_train)
    lr_best_params = {k.replace("clf__", ""): v for k, v in lr_search.best_params_.items()}
    print(f"Best Logistic Regression params: {lr_best_params}")
    print(f"Best Logistic Regression CV ROC-AUC: {lr_search.best_score_:.4f}")

    result = {
        "n_splits": args.n_splits,
        "purge_gap": FORWARD_WINDOW,
        "xgboost": {"params": xgb_best_params, "cv_roc_auc": xgb_search.best_score_},
        "logistic_regression": {"params": lr_best_params, "cv_roc_auc": lr_search.best_score_},
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "tuned_hyperparameters.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved tuned hyperparameters -> {out_path}")
    print("Next: run 05_train_model.py -- it will pick these up automatically, "
          "then evaluate the RESULT on val (never used in this search).")


if __name__ == "__main__":
    main()
