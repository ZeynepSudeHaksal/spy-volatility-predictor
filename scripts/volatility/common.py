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
from sklearn.metrics import accuracy_score, f1_score, precision_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "volatility"
MODELS_DIR = PROJECT_ROOT / "models" / "volatility"

BASKET = ["SPY", "QQQ", "IWM", "DIA", "XLF", "GLD", "TLT"]

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
    labels/probabilities -- subject to a precision guardrail (see
    below) that rules out degenerate near-constant-prediction
    thresholds.

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
    useless at the one thing a risk-flagging tool needs to do.

    THE PRECISION GUARDRAIL -- added after GLD/TLT were added to the
    basket: plain F1-maximization has the SAME failure mode from the
    opposite direction. When the validation set's class balance shifts
    (adding non-equity assets changed the high-vol/normal mix), F1 can
    be maximized by dragging the threshold low enough that the model
    predicts "high" on almost every day (recall near 100%) -- this
    inflates F1 even though the predictions carry almost no
    information. Concretely, this happened: thresholds of 0.35-0.40
    produced ~95-99% recall and accuracy well below the naive
    baseline, across every ticker including ones that worked fine
    before.

    The fix: a threshold is only eligible if its precision beats the
    base rate (the actual prevalence of the high-vol class in this
    data) by a MEANINGFUL margin, not just technically. A model
    predicting "high" nearly everywhere has precision that converges
    to exactly the base rate -- it's doing no better than guessing
    positive at random with the right frequency. A strict ">" isn't
    enough of a guard on its own: in the run that motivated this fix,
    the degenerate threshold's validation precision (0.364) came in
    almost exactly equal to the base rate (0.363) -- close enough that
    ordinary sampling noise could tip a bare ">" either way. PRECISION_LIFT
    requires precision to beat the base rate by a fixed relative
    margin (15%), which is a real "doing meaningfully better than
    chance" bar rather than a coin flip against noise. This rules out
    both degenerate directions (threshold too high -> recall gutted;
    threshold too low -> precision no better than chance) while still
    letting F1 pick the best threshold among the thresholds that are
    actually discriminating.

    If NO threshold clears the guardrail (possible on a small or
    unusually skewed validation split), we fall back to unconstrained
    F1 selection and print a warning -- that's a sign this
    ticker/split needs a closer look, not something to silently paper
    over.

    Ties (within the eligible set) are broken by preferring the
    threshold closest to 0.5.
    """
    PRECISION_LIFT = 1.15  # precision must beat base rate by >= 15% relative

    candidates = np.round(np.arange(0.05, 0.96, 0.01), 2)
    base_rate = float(np.mean(y_true))  # prevalence of the high-vol class
    min_precision = base_rate * PRECISION_LIFT

    scored = []
    for t in candidates:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        scored.append((t, f1, precision))

    eligible = [(t, f1) for t, f1, p in scored if p >= min_precision]
    if not eligible:
        print(
            "WARNING: select_f1_threshold found no threshold with "
            f"precision >= {PRECISION_LIFT}x the base rate ({base_rate:.3f}) "
            "-- falling back to unconstrained F1 selection. This usually means the "
            "validation split is small or unusually skewed; check this "
            "ticker/run's numbers carefully rather than trusting them "
            "blindly.",
            file=sys.stderr,
        )
        eligible = [(t, f1) for t, f1, _ in scored]

    best_f1 = max(f1 for _, f1 in eligible)
    best_thresholds = [t for t, f1 in eligible if f1 == best_f1]
    return float(min(best_thresholds, key=lambda t: abs(t - 0.5)))


MIN_ROWS_FOR_OWN_THRESHOLD = 60  # below this, borrow the pooled threshold
                                   # rather than tune an unreliable one


def select_f1_thresholds_per_ticker(
    y_true, y_proba, tickers, basket: list[str]
) -> dict[str, float]:
    """
    Picks a SEPARATE select_f1_threshold() cutoff for each ticker in
    the basket, instead of one pooled threshold shared across all of
    them.

    WHY: one pooled model trained across the whole basket is fine --
    it shares what it learns about volatility clustering while still
    picking up ticker-specific effects through the ticker dummy
    features. But a single pooled DECISION THRESHOLD assumes every
    ticker's volatility dynamics are similar enough to share one
    cutoff, and they aren't. This is exactly what broke when GLD
    (gold) and TLT (long Treasuries) were added alongside the
    original equity-index ETFs: a pooled threshold tuned to satisfy
    the precision guardrail across all seven tickers at once ended up
    far too strict for XLF and TLT specifically -- their test recall
    collapsed to ~10% and ~2% respectively (the model almost never
    flagged either as high-vol) even though the POOLED metrics looked
    fine. Averaging across tickers hid a real per-ticker failure.
    Per-ticker thresholds fix this directly: each ticker gets the cut
    that actually works for its own volatility distribution.

    A ticker with too few validation rows to tune its own threshold
    reliably (< MIN_ROWS_FOR_OWN_THRESHOLD) borrows the pooled
    threshold instead -- better to use a reasonable shared number than
    overfit a cutoff to a handful of rows.
    """
    pooled_fallback = select_f1_threshold(y_true, y_proba)

    thresholds = {}
    ticker_mask_source = tickers.values if hasattr(tickers, "values") else tickers
    for ticker in basket:
        mask = ticker_mask_source == ticker
        n = int(mask.sum())
        if n < MIN_ROWS_FOR_OWN_THRESHOLD:
            print(
                f"NOTE: {ticker} has only {n} validation rows -- using the "
                f"pooled threshold ({pooled_fallback:.2f}) rather than tuning "
                "its own (too few rows to trust a ticker-specific cut).",
                file=sys.stderr,
            )
            thresholds[ticker] = pooled_fallback
            continue
        thresholds[ticker] = select_f1_threshold(y_true[mask], y_proba[mask])
    return thresholds


def apply_per_ticker_thresholds(y_proba, tickers, thresholds: dict[str, float]) -> np.ndarray:
    """
    Converts probabilities to 0/1 predictions using each row's OWN
    ticker's threshold (from select_f1_thresholds_per_ticker), rather
    than one threshold applied uniformly to every row. Shared here so
    05_train_model.py (val) and 06_evaluate.py (test) apply thresholds
    identically.
    """
    ticker_values = tickers.values if hasattr(tickers, "values") else tickers
    cuts = np.array([thresholds[t] for t in ticker_values])
    return (np.asarray(y_proba) >= cuts).astype(int)
