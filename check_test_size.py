import pandas as pd

test = pd.read_csv('data/processed/direction/test.csv', index_col='Date', parse_dates=True)

print(f"Total test rows: {len(test):,}")
print(f"Date range: {test.index.min().date()} to {test.index.max().date()}\n")

print("Per-ticker test row counts:")
for ticker, group in test.groupby('ticker'):
    n = len(group)
    # rough standard error on ROC-AUC for a roughly balanced binary
    # outcome at this sample size
    se_estimate = (0.25 / n) ** 0.5
    print(f"  {ticker:>4}: {n:>4} rows   (rough ROC-AUC noise band: +/- {se_estimate:.3f})")
