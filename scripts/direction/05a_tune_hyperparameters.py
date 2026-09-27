"""
05a_tune_hyperparameters.py

Purged walk-forward hyperparameter search for the direction pipeline.
See scripts/volatility/05a_tune_hyperparameters.py for the full
rationale (identical here) -- purged walk-forward CV via
scripts/cv_utils.py, searched on TRAIN only, val/test untouched.

Usage:
    python scripts/direction/05a_tune_hyperparameters.py
    python scripts/direction/05a_tune_hyperparameters.py --n-splits 5 --n-iter 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # for cv_utils
from cv_utils import purged_walk_forward_splits

from common import LABEL_COL, MODELS_DIR, build_feature_matrix, load_split

FORWARD_WINDOW = 5  # must match 03_label_data.py

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
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=25)
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
    print("Next: run 05_train_model.py -- it will pick these up automatically.")


if __name__ == "__main__":
    main()
