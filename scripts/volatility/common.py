"""
common.py

Shared definitions for the volatility pipeline's training and tuning
scripts (05_train_model.py, 05a_tune_hyperparameters.py). Pulled out
here so both scripts build the exact same feature matrix from the same
column list, the same way, every time -- a tuning script that quietly
drifted from the training script's feature-building logic would
invalidate every hyperparameter it found.

CRITICAL: LEAKAGE GUARD -- labeled_data.csv contains forward_vol_5d and
trailing_vol_median_60d, used to BUILD the label, never as model
inputs. NUMERIC_FEATURE_COLS is an explicit allow-list for exactly
this reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "volatility"
MODELS_DIR = PROJECT_ROOT / "models" / "volatility"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF"]

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


def select_f1_threshold(y_true, y_proba) -> float:
    """
    Picks the probability threshold (0.05-0.95, step 0.01) that
    maximizes F1 on the high-vol class, on the given (val)
    labels/probabilities.

    WHY F1, NOT ACCURACY: predict_proba() >= 0.5 is an arbitrary
    default, not something that's guaranteed to produce good accuracy
    -- especially with class_weight="balanced" (LR) / scale_pos_weight
    (XGBoost), both of which deliberately shift probabilities toward
    predicting the minority (high-vol) class more often, which can
    make raw accuracy at 0.5 come in BELOW the naive "always predict
    the majority class" baseline even when the model's probabilities
    are clearly informative (see ROC-AUC).

    An earlier version of this function maximized accuracy directly --
    don't do that. On this imbalanced a problem (roughly 40/60
    high-vol/normal), accuracy is maximized by pushing the threshold
    so high the model barely ever predicts high-vol at all: in testing
    this produced a threshold where XGBoost's test recall collapsed to
    ~6% (0% on XLF specifically -- it never predicted high-vol for that
    ticker once), i.e. accuracy went up by making the model nearly
    useless at the one thing a risk-flagging tool needs to do. F1
    balances precision and recall, so it won't let accuracy's
    "just predict the majority class more" shortcut win -- it still
    moves the threshold up from 0.5 toward better precision, just not
    so far that recall is gutted.

    Ties are broken by preferring the threshold closest to 0.5, so we
    don't drift into a degenerate near-constant-prediction threshold.
    """
    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [
        f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        for t in candidates
    ]
    best_f1 = max(f1s)
    best_thresholds = [t for t, f in zip(candidates, f1s) if f == best_f1]
    return float(min(best_thresholds, key=lambda t: abs(t - 0.5)))
