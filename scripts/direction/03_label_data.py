"""
03_label_data.py

Builds the classification target: for each (ticker, date), was the
CUMULATIVE return over the next 5 trading days positive (1, "up") or
not (0, "down/flat")?

forward_return_5d(t) = Close[t+5] / Close[t] - 1
label = 1 if forward_return_5d > 0 else 0

SAME SAFE PATTERN AS THE VOLATILITY PROJECT, AND FOR THE SAME REASON
--------------------------------------------------------------------------
features.csv is a long panel -- every ticker shares the same Date index,
so dates are duplicated across rows. A prior version of the companion
volatility project discovered (and fixed) a real bug where
`df.groupby('ticker').apply(...)` assigned back onto a duplicate-indexed
frame silently SWAPPED values between tickers on shared dates. The fix
carried over here: reset to a genuinely unique row index before any
groupby+apply that returns a per-row Series, then restore the Date
index afterward for a readable file. See that project's 03_label_data.py
for the full writeup and the verification that caught it.

WHY THE BASELINE MATTERS MORE HERE THAN IN THE VOLATILITY PROJECT
------------------------------------------------------------------------
Equity indices have historical positive drift -- SPY has been "up" over
a given forward window well more than 50% of the time across most
multi-year stretches. That means "always predict up" is a NON-TRIVIAL
baseline, often in the high 50s%, not a 50/50 coin flip. A direction
model has to beat THAT, not 50%, to be worth anything. This script
prints the up-rate per ticker so that baseline is visible from the very
first step, not just in evaluation.

Usage:
    python scripts/03_label_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "direction"

FORWARD_WINDOW = 5


def load_features() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "features.csv"
    if not path.exists():
        print("ERROR: features.csv not found. Run 02_engineer_features.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def main():
    df = load_features()

    # --- SAFE PATTERN: unique row id before any groupby+apply realignment ---
    df = df.reset_index()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    grouped = df.groupby("ticker", group_keys=False)
    df["forward_return_5d"] = grouped["Close"].apply(
        lambda s: s.shift(-FORWARD_WINDOW) / s - 1
    )
    # --- end safe pattern ---

    df["label"] = (df["forward_return_5d"] > 0).astype(int)

    n_before = len(df)
    df = df.dropna(subset=["forward_return_5d"])
    n_after = len(df)
    print(f"Dropped {n_before - n_after} rows without a valid forward window (per ticker).")

    df = df.sort_values(["Date", "ticker"]).set_index("Date")

    out_path = PROCESSED_DATA_DIR / "labeled_data.csv"
    df.to_csv(out_path)
    print(f"Saved {len(df):,} labeled rows -> {out_path.relative_to(PROJECT_ROOT)}")

    print("\nUp-rate by ticker (this IS the 'always predict up' baseline accuracy):")
    for ticker, group in df.groupby("ticker"):
        up_pct = (group["label"] == 1).mean() * 100
        print(f"  {ticker:>4}: {len(group):>5,} rows | up {up_pct:5.1f}% of the time")

    overall_up_pct = (df["label"] == 1).mean() * 100
    print(f"\nPooled up-rate: {overall_up_pct:.1f}% -- any model needs to clear THIS "
          f"bar, not 50%, to demonstrate real skill.")

    print("\nDone. Next: run 04_train_test_split.py")


if __name__ == "__main__":
    main()
