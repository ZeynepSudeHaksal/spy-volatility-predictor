import pandas as pd
import numpy as np

numeric_cols = [
    'momentum_5d', 'momentum_10d', 'momentum_20d',
    'rsi_14', 'macd', 'macd_hist',
    'price_vs_ma20', 'price_vs_ma50',
    'volume_ratio', 'realized_vol_20d',
    'vix_level', 'vix_spy_spread', 'log_return',
]

train = pd.read_csv('data/processed/direction/train.csv', index_col='Date', parse_dates=True)

print('Std dev of each feature in TRAIN (StandardScaler divides by this):')
for col in numeric_cols:
    std = train[col].std()
    flag = '  <-- NEAR ZERO, WILL BREAK SCALER' if std < 1e-8 else ''
    print(f'  {col:>18}: std={std:.8f}{flag}')

print('\nTicker dummy variance check:')
dummies = pd.get_dummies(train['ticker'], prefix='ticker')
for col in dummies.columns:
    std = dummies[col].astype(float).std()
    flag = '  <-- NEAR ZERO' if std < 1e-8 else ''
    print(f'  {col:>18}: std={std:.8f}{flag}')

print('\nAny NaN in feature matrix after building it the same way common.py does:')
X = train[numeric_cols]
X_full = pd.concat([X, dummies], axis=1)
print(f'  Total NaN count: {X_full.isna().sum().sum()}')
print(f'  Total Inf count: {np.isinf(X_full.select_dtypes(include=[np.number])).sum().sum()}')
