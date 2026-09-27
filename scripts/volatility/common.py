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

import pandas as pd

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
