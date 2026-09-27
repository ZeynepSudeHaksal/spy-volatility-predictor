import pandas as pd
import numpy as np

for split in ['train', 'val', 'test']:
    df = pd.read_csv(f'data/processed/direction/{split}.csv', index_col='Date', parse_dates=True)
    numeric_cols = [
        'momentum_5d', 'momentum_10d', 'momentum_20d',
        'rsi_14', 'macd', 'macd_hist',
        'price_vs_ma20', 'price_vs_ma50',
        'volume_ratio', 'realized_vol_20d',
        'vix_level', 'vix_spy_spread', 'log_return',
    ]
    print(f'--- {split} ---')
    found_issue = False
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        n_nan = df[col].isna().sum()
        col_max = df[col].abs().max()
        if n_inf > 0 or n_nan > 0 or col_max > 1000:
            print(f'  {col}: inf={n_inf} nan={n_nan} max_abs={col_max:.2f}')
            found_issue = True
    if not found_issue:
        print('  (nothing extreme found)')
