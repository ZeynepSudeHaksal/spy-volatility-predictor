"""
03_label_data.py

Builds the classification target: for each (ticker, date), was the
forward-looking 5-day realized volatility "high" (1) or "low" (0)
relative to that ticker's own trailing 60-day median?

EVERYTHING HERE IS GROUPED BY TICKER
---------------------------------------
Both the forward-vol calculation and the trailing-median baseline are
computed within each ticker's own series (via groupby('ticker')). A
regime is also defined RELATIVE TO EACH TICKER'S OWN HISTORY, not a
basket-wide threshold -- IWM (small-cap) is structurally more volatile
than DIA (large-cap blue-chip) on an absolute basis, and "high vol for
IWM" should mean "elevated relative to IWM's own normal."

A SHARP PANDAS PITFALL THIS SCRIPT DELIBERATELY AVOIDS
-----------------------------------------------------------
features.csv is a long panel: every ticker shares the same Date index,
so DATES ARE DUPLICATED across rows (5 tickers -> 5 rows per date).
Doing `df.groupby('ticker')['col'].apply(fn)` and assigning the result
straight back with `df['new_col'] = result` is DANGEROUS here: pandas
aligns Series assignment by index label, group-apply does not
guarantee the result comes back in the original row order, and with
DUPLICATE labels that combination can silently swap values between
tickers on rows that share a date -- with no error, no warning, just
wrong numbers. This was caught and hand-verified during development
(see the project notes) before it reached the model.

The fix: before any groupby+apply that returns a per-row Series, reset
to a genuinely UNIQUE row index. compute_forward_realized_vol and the
rolling median both preserve whatever index they're given, so with a
unique index the group-apply result realigns back onto the original
frame correctly regardless of concatenation order. The Date index is
restored afterward, purely for a readable output file.

Label: 1 = "high vol regime" (forward_vol_5d > that ticker's own
             trailing median),
       0 = "low vol regime" (forward_vol_5d <= trailing median)

Usage:
    python scripts/03_label_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

FORWARD_WINDOW = 5
TRAILING_MEDIAN_WINDOW = 60
TRADING_DAYS_PER_YEAR = 252


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features.csv"
    if not path.exists():
        print("ERROR: features.csv not found. Run 02_engineer_features.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def compute_forward_realized_vol(log_return: pd.Series, window: int) -> pd.Series:
    """For each row, annualized std of log returns at t+1, ..., t+window.
    Preserves log_return's index -- required for the safe-realignment
    pattern described in the module docstring."""
    future_returns = pd.concat(
        [log_return.shift(-i) for i in range(1, window + 1)], axis=1
    )
    return future_returns.std(axis=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def main():
    df = load_features()

    # --- SAFE PATTERN: unique row id before any groupby+apply realignment ---
    df = df.reset_index()  # 'Date' becomes a plain column
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)  # unique RangeIndex

    grouped = df.groupby("ticker", group_keys=False)
    df["forward_vol_5d"] = grouped["log_return"].apply(
        lambda s: compute_forward_realized_vol(s, FORWARD_WINDOW)
    )
    df["trailing_vol_median_60d"] = grouped["realized_vol_20d"].apply(
        lambda s: s.rolling(TRAILING_MEDIAN_WINDOW).median()
    )
    # --- end safe pattern ---

    df["label"] = (df["forward_vol_5d"] > df["trailing_vol_median_60d"]).astype(int)

    n_before = len(df)
    df = df.dropna(subset=["forward_vol_5d", "trailing_vol_median_60d"])
    n_after = len(df)
    print(f"Dropped {n_before - n_after} rows without a valid label "
          f"(no forward window and/or insufficient trailing history, per ticker).")

    # Restore Date as the index and chronological (date, ticker) order for
    # a readable output file.
    df = df.sort_values(["Date", "ticker"]).set_index("Date")

    out_path = PROCESSED_DATA_DIR / "labeled_data.csv"
    df.to_csv(out_path)

    print(f"Saved {len(df):,} labeled rows -> {out_path.relative_to(PROJECT_ROOT)}")

    print("\nClass balance by ticker:")
    for ticker, group in df.groupby("ticker"):
        counts = group["label"].value_counts().sort_index()
        high_pct = counts.get(1, 0) / len(group) * 100
        print(f"  {ticker:>4}: {len(group):>5,} rows | high-vol regime: {high_pct:5.1f}%")

    print("\nDone. Next: run 04_train_test_split.py")


if __name__ == "__main__":
    main()
