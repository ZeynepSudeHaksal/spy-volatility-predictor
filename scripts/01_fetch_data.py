"""
01_fetch_data.py

Pulls raw daily OHLCV data for a basket of index/sector ETFs, plus the
VIX index, from Yahoo Finance, and saves them to data/raw/ as CSVs.

Basket: SPY, QQQ, IWM, DIA, XLF -- broad index ETFs (S&P 500, Nasdaq
100, Russell 2000, Dow 30) plus one sector fund (Financials), rather
than individual stocks. Volatility regime is fundamentally a
market-wide phenomenon, so staying with index/sector ETFs keeps the
project's thesis intact instead of drifting into single-stock noise
(earnings surprises, idiosyncratic news) that a broad regime model
isn't meant to capture.

Why VIX is pulled once, separately, rather than per-ticker:
VIX is the market's own forward-looking volatility estimate (derived
from S&P 500 options), and it's used as a shared market-wide feature
for every ticker in the basket -- not something to re-fetch per ticker.

Usage:
    python scripts/01_fetch_data.py
    python scripts/01_fetch_data.py --start 2005-01-01 --end 2026-09-19
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Project root is one level up from this script's directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF"]
VIX_SYMBOL = "^VIX"

DEFAULT_START = "2005-01-01"  # far enough back to span multiple vol regimes
                                # (dot-com aftermath, '08 GFC, '18 vol spike,
                                # '20 COVID crash, '22 rate-hike selloff)


def fetch_ticker(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker via yfinance."""
    print(f"Fetching {symbol} from {start} to {end or 'today'}...")
    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts OHLC for splits/dividends
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No data returned for {symbol}. Check the ticker symbol, "
            f"date range, or your network connection."
        )

    # yfinance sometimes returns MultiIndex columns even for a single
    # ticker -- flatten them so downstream scripts get plain column names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch raw SPY + VIX data.")
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start date (YYYY-MM-DD). Default: {DEFAULT_START}",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (YYYY-MM-DD). Default: today.",
    )
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_symbols = {ticker: ticker for ticker in BASKET}
    all_symbols["VIX"] = VIX_SYMBOL

    for label, symbol in all_symbols.items():
        try:
            df = fetch_ticker(symbol, args.start, args.end)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        out_path = RAW_DATA_DIR / f"{label.lower()}_raw.csv"
        df.to_csv(out_path)
        print(f"  Saved {len(df):,} rows -> {out_path.relative_to(PROJECT_ROOT)}")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")

    print(f"\nBasket: {', '.join(BASKET)} (+ VIX)")
    print("Done. Next: run 02_engineer_features.py")


if __name__ == "__main__":
    main()
