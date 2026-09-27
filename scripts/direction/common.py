"""
common.py

Shared definitions for the direction pipeline's training and tuning
scripts (05_train_model.py, 05a_tune_hyperparameters.py). See
scripts/volatility/common.py for the full rationale -- same idea,
mirrored here so the two pipelines don't have to import across
directories.

CRITICAL: LEAKAGE GUARD -- forward_return_5d is the value the label
was built from and must never be a model input. NUMERIC_FEATURE_COLS
is an explicit allow-list for exactly this reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

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
    "vol_regime_probability",  # from 02b_add_volatility_regime.py --
    # requires that script to have been run (see its docstring for the
    # honest caveat on how to interpret this feature)
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


def build_feature_matrix(df: pd.DataFrame, ticker_dummy_cols: list[str] | None, numeric_cols: list[str] | None = None):
    """
    Builds [numeric features] + [one-hot ticker dummies]. If
    ticker_dummy_cols is given (from train), the dummy columns are
    reindexed to exactly match it -- so val/test/live data always
    produces a feature matrix with the same shape and column order the
    model was trained on, even if a ticker is missing from that split.

    numeric_cols defaults to the module-level NUMERIC_FEATURE_COLS, but
    can be overridden (e.g. by 05_train_model.py's --exclude-features)
    for ablation studies without duplicating this function.
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_FEATURE_COLS

    numeric = df[numeric_cols]
    dummies = pd.get_dummies(df["ticker"], prefix="ticker")
    if ticker_dummy_cols is not None:
        dummies = dummies.reindex(columns=ticker_dummy_cols, fill_value=0)
        cols_out = ticker_dummy_cols
    else:
        cols_out = list(dummies.columns)
    X = pd.concat([numeric, dummies], axis=1)
    return X, cols_out