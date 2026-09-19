"""
02_engineer_features.py

Builds the feature set for the volatility regime classifier from the raw
per-ticker CSVs and the shared VIX CSV produced by 01_fetch_data.py.

BASKET / PANEL DESIGN
------------------------
This produces a long-format panel: one row per (ticker, date), all
tickers stacked into a single features.csv with a 'ticker' column.
Every feature below is computed WITHIN each ticker's own series (via
groupby('ticker')) before stacking -- a rolling window must never blend
returns from two different tickers together, even though they share a
trading calendar. VIX is the one feature that's genuinely shared across
tickers (it's a single market-wide series), so it's joined in after the
per-ticker features are computed.

Feature groups (and why each one is here):

1. Realized volatility (multiple lookback windows)
   The single most important input. Volatility clusters over time, so
   *recent* realized vol is the strongest predictor of *near-future*
   realized vol. Using three windows (5/10/20-day) instead of one lets
   the model see both fast-moving and slower-moving vol trends.

2. ATR (Average True Range)
   Captures intraday range that close-to-close returns miss -- a day
   that opens, spikes, and closes flat looks calm in close-to-close
   terms but was not calm to trade through.

3. Volume relative to its own moving average
   A *relative* measure (today's volume / 20-day average) rather than
   raw volume, so it stays meaningful both across tickers of very
   different sizes (SPY vs. XLF) and across decades of rising volume.

4. VIX level and VIX-SPY-style spread
   VIX is the market's own forward-looking volatility estimate. For
   basket tickers other than SPY, the "spread" is still computed
   against that ticker's OWN realized vol (VIX minus ticker_realized_vol_20d)
   -- this captures whether the market's broad fear gauge is running
   hot relative to what THIS particular ticker's own recent moves would
   suggest, which is meaningful even for QQQ/IWM/DIA/XLF, not just SPY.

5. Return magnitude (not direction)
   |daily return| is a simple, direct volatility proxy at the
   single-day level.

Usage:
    python scripts/02_engineer_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF"]

REALIZED_VOL_WINDOWS = [5, 10, 20]   # trading days
ATR_WINDOW = 14                       # standard ATR lookback
VOLUME_MA_WINDOW = 20
TRADING_DAYS_PER_YEAR = 252


def load_ticker_raw(ticker: str) -> pd.DataFrame:
    path = RAW_DATA_DIR / f"{ticker.lower()}_raw.csv"
    if not path.exists():
        print(
            f"ERROR: {path.name} not found. Run 01_fetch_data.py first.",
            file=sys.stderr,
        )
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def load_vix() -> pd.DataFrame:
    path = RAW_DATA_DIR / "vix_raw.csv"
    if not path.exists():
        print("ERROR: vix_raw.csv not found. Run 01_fetch_data.py first.",
              file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def engineer_single_ticker(df: pd.DataFrame, vix: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """All per-ticker feature computation -- no cross-ticker mixing here."""
    df = df.copy()

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["abs_return"] = df["log_return"].abs()

    for window in REALIZED_VOL_WINDOWS:
        df[f"realized_vol_{window}d"] = (
            df["log_return"].rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    prev_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = true_range.rolling(ATR_WINDOW).mean()
    df["atr_pct"] = df["atr"] / df["Close"]

    volume_ma = df["Volume"].rolling(VOLUME_MA_WINDOW).mean()
    df["volume_ratio"] = df["Volume"] / volume_ma

    # Join VIX (shared across all tickers), scale to fractional (0.20 = 20%)
    merged = df.join(vix[["Close"]].rename(columns={"Close": "vix_level"}), how="inner")
    merged["vix_level"] = merged["vix_level"] / 100
    merged["vix_spy_spread"] = merged["vix_level"] - merged["realized_vol_20d"]

    merged["ticker"] = ticker
    return merged


def main():
    vix = load_vix()

    feature_cols = [
        "log_return", "abs_return",
        "realized_vol_5d", "realized_vol_10d", "realized_vol_20d",
        "atr", "atr_pct",
        "volume_ratio",
        "vix_level", "vix_spy_spread",
    ]

    all_rows = []
    for ticker in BASKET:
        raw = load_ticker_raw(ticker)
        engineered = engineer_single_ticker(raw, vix, ticker)
        subset = engineered[["ticker", "Close"] + feature_cols].copy()

        n_before = len(subset)
        subset = subset.dropna()
        n_after = len(subset)
        print(f"{ticker}: dropped {n_before - n_after} warmup rows, "
              f"kept {n_after:,} rows "
              f"({subset.index.min().date()} to {subset.index.max().date()})")

        all_rows.append(subset)

    # Stack all tickers, then sort by date then ticker so the file reads
    # naturally and downstream groupby('ticker') operations are tidy.
    panel = pd.concat(all_rows)
    panel = panel.reset_index().sort_values(["Date", "ticker"]).set_index("Date")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / "features.csv"
    panel.to_csv(out_path)

    print(f"\nSaved {len(panel):,} total rows ({len(BASKET)} tickers x ~dates) "
          f"-> {out_path.relative_to(PROJECT_ROOT)}")
    print("\nRows per ticker:")
    print(panel["ticker"].value_counts())
    print("\nDone. Next: run 03_label_data.py")


if __name__ == "__main__":
    main()
