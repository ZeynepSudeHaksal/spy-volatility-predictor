"""
promote_if_safe.py

The safety gate for the WEEKLY automated retrain (see
.github/workflows/weekly-retrain.yml). Runs after the full pipeline
(01-06) has already retrained and overwritten models/volatility/
metadata.json -- compares the fresh run's metrics against the metrics
that were actually live in production before this run started, and
exits non-zero if the new model looks meaningfully worse.

WHY THIS EXISTS: this project already found, the hard way, that an
unreviewed process can silently ship a regression -- adding GLD/TLT to
the basket broke the (then-pooled) decision threshold in a way that
looked fine in the pooled metrics while individual tickers' recall
collapsed to near zero. A human caught that only because a human was
looking at per-ticker numbers after every run. A weekly job that
retrains and deploys with NO ONE watching needs an automated stand-in
for that same scrutiny, or the exact same failure mode can reappear
and go live with nobody noticing until a user does.

WHAT THIS CHECKS (XGBoost only -- it's the model actually served, per
metadata["production_model"] / 06_evaluate.py):

1. Pooled ROC-AUC clears an absolute floor (the model has to be doing
   something better than a coin flip, full stop).
2. Pooled ROC-AUC hasn't dropped too far from the previous run (not
   just "still above the floor" -- a real regression).
3. EVERY ticker's recall clears an absolute floor. This is the direct
   regression test for the bug that motivated this script: XLF and TLT
   both silently fell to ~10%/~2% recall under a prior pooled-threshold
   design while pooled metrics still looked acceptable.
4. EVERY ticker present in both the old and new run hasn't had its
   recall drop too far from where it was.

If there's no previous metadata.json to compare against (first run
ever, or it was deleted), only the absolute floors (1 and 3) apply --
there's nothing to regress against yet.

Usage:
    python scripts/volatility/promote_if_safe.py \\
        --new models/volatility/metadata.json \\
        --old /tmp/old_metadata.json   # optional; omit if it doesn't exist
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# -- Thresholds. Chosen to be strict enough to catch the kind of failure
# this project already hit once, loose enough not to block normal
# week-to-week variation in a walk-forward-evaluated model. --
MIN_POOLED_ROC_AUC = 0.55
MAX_POOLED_ROC_AUC_DROP = 0.05
MIN_TICKER_RECALL = 0.10
MAX_TICKER_RECALL_DROP = 0.15


def load_metadata(path: Path) -> dict | None:
    if path is None or not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def check(label: str, passed: bool, detail: str, failures: list[str]) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {label}: {detail}")
    if not passed:
        failures.append(f"{label}: {detail}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--old", required=False, type=Path, default=None)
    args = parser.parse_args()

    new = load_metadata(args.new)
    if new is None:
        print(f"ERROR: {args.new} not found -- did the retrain pipeline "
              "actually run?", file=sys.stderr)
        sys.exit(1)
    old = load_metadata(args.old)

    failures: list[str] = []

    print("Checking the new run's pooled XGBoost metrics...")
    new_pooled = new["test_metrics_pooled"]["xgboost"]
    new_roc_auc = new_pooled["roc_auc"]

    check(
        "pooled ROC-AUC floor",
        new_roc_auc >= MIN_POOLED_ROC_AUC,
        f"{new_roc_auc:.4f} (need >= {MIN_POOLED_ROC_AUC})",
        failures,
    )

    if old is not None:
        old_pooled = old["test_metrics_pooled"]["xgboost"]
        old_roc_auc = old_pooled["roc_auc"]
        drop = old_roc_auc - new_roc_auc
        check(
            "pooled ROC-AUC regression",
            drop <= MAX_POOLED_ROC_AUC_DROP,
            f"{old_roc_auc:.4f} -> {new_roc_auc:.4f} "
            f"(dropped {drop:.4f}, allowed <= {MAX_POOLED_ROC_AUC_DROP})",
            failures,
        )
    else:
        print("  [SKIP] pooled ROC-AUC regression: no previous metadata.json "
              "to compare against")

    print("\nChecking every ticker's XGBoost recall...")
    new_by_ticker = new["test_metrics_by_ticker"]
    old_by_ticker = old["test_metrics_by_ticker"] if old is not None else {}

    for ticker, metrics in new_by_ticker.items():
        recall = metrics["recall"]
        check(
            f"{ticker} recall floor",
            recall >= MIN_TICKER_RECALL,
            f"{recall:.3f} (need >= {MIN_TICKER_RECALL})",
            failures,
        )

        if ticker in old_by_ticker:
            old_recall = old_by_ticker[ticker]["recall"]
            drop = old_recall - recall
            check(
                f"{ticker} recall regression",
                drop <= MAX_TICKER_RECALL_DROP,
                f"{old_recall:.3f} -> {recall:.3f} "
                f"(dropped {drop:.3f}, allowed <= {MAX_TICKER_RECALL_DROP})",
                failures,
            )

    print()
    if failures:
        print(f"REJECTED: {len(failures)} check(s) failed. The currently-"
              "deployed model will NOT be replaced -- nothing gets "
              "committed or pushed this run.")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("APPROVED: all checks passed. Safe to commit and deploy this run.")
    sys.exit(0)


if __name__ == "__main__":
    main()
