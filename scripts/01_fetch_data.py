"""
01_fetch_data.py

Pulls raw daily OHLCV data for a basket of index/sector ETFs, plus the
VIX index, from Yahoo Finance, and saves them to data/raw/ as CSVs.

SHARED by both prediction pipelines in this project (scripts/volatility/
and scripts/direction/) -- the raw data pull is identical either way, so
it lives once at the top level rather than being duplicated per pipeline.

Basket: SPY, QQQ, IWM, DIA, XLF, GLD, TLT -- broad index ETFs (S&P 500,
Nasdaq 100, Russell 2000, Dow 30), one sector fund (Financials), and
two non-equity asset classes (Gold, long-term Treasuries) added to see
how the models generalize outside pure equities, rather than
individual stocks.

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

# Make scripts/ importable regardless of the CWD this is invoked from
# (python scripts/01_fetch_data.py already adds scripts/ to sys.path, but
# being explicit here means this also works if something imports this
# module rather than running it directly).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_utils import BASKET, DEFAULT_START, VIX_SYMBOL, fetch_ticker

# Project root is one level up from this script's directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


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
