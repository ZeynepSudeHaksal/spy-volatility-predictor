# Market Analytics (Volatility Regime + Direction Prediction)

Two ML prediction modes for a basket of index/sector ETFs (SPY, QQQ, IWM,
DIA, XLF), sharing one data pipeline foundation, one Flask backend, and
one React dashboard with a top-level mode toggle:

- **Volatility regime**: will 5-day forward realized volatility be
  elevated or normal, relative to each ticker's own recent history?
  A risk-flagging tool for portfolio triage.
- **Direction**: will price be up or down 5 trading days out? A
  deliberately harder, more honestly-framed problem -- see below.

Both are pooled models (one XGBoost model per mode, `ticker` as a
one-hot feature, trained across the whole basket), explained with SHAP,
evaluated on a chronologically held-out test period with a per-ticker
purge gap to prevent label lookahead leakage.

## Why two separate pipelines, one shared foundation

The raw data pull (`scripts/01_fetch_data.py`) is identical either way
-- same basket, same VIX -- so it's shared at the top level. Everything
downstream diverges: different features, different labels, different
models, different evaluation criteria, because volatility clustering
and price direction are fundamentally different prediction problems
with different signal structure. Keeping them as separate pipelines
under `scripts/volatility/` and `scripts/direction/` avoids forcing a
false symmetry between two things that aren't actually alike.

## Why direction is the harder problem

Volatility clusters -- recent volatility is a strong, direct predictor
of near-future volatility. Price direction doesn't work that way:
liquid index ETFs are close to efficiently priced at a 5-day horizon,
and published research on this kind of task typically lands in the
low-to-mid 50s% accuracy. The direction pipeline is built to represent
that honestly: every evaluation step reports model accuracy against an
**"always predict up" baseline** (reflecting real historical equity
drift, often in the high 50s%), not against a 50% coin flip, because
that's the bar that actually demonstrates skill.

## Project layout

```
├── scripts/
│   ├── 01_fetch_data.py        # SHARED: basket + VIX pull
│   ├── volatility/              # 02-06: features, labels, split, train, eval
│   └── direction/                # 02-06: features, labels, split, train, eval
├── data/
│   ├── raw/                      # shared
│   └── processed/{volatility,direction}/
├── models/{volatility,direction}/
├── backend/
│   ├── app.py                    # one Flask app, /api/volatility/* and /api/direction/*
│   └── data/{volatility,direction}/
└── frontend/
    └── src/
        ├── App.jsx               # mode toggle -> ticker tabs -> mode-specific view
        └── components/
            ├── ModeToggle.jsx, TickerSelector.jsx   # shared
            ├── volatility/        # RegimeHero, FeatureBreakdown, HistoryChart
            └── direction/          # DirectionHero, FeatureBreakdown, HistoryChart
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python scripts/01_fetch_data.py          # shared, run once

python scripts/volatility/02_engineer_features.py
python scripts/volatility/03_label_data.py
python scripts/volatility/04_train_test_split.py
python scripts/volatility/05_train_model.py
python scripts/volatility/06_evaluate.py

python scripts/direction/02_engineer_features.py
python scripts/direction/03_label_data.py
python scripts/direction/04_train_test_split.py
python scripts/direction/05_train_model.py
python scripts/direction/06_evaluate.py
```

```bash
cd backend && pip install -r requirements.txt && python app.py   # :5001
```

```bash
cd frontend && npm install && npm run dev   # :5173
```

Open the dashboard, use the **Volatility / Direction** toggle at the top
to switch modes, and the ticker tabs underneath to switch tickers within
whichever mode is active.

## Roadmap

- [x] Shared data pipeline + two independent prediction pipelines
- [x] Walk-forward split, per-ticker purge gaps, leakage guards (both modes)
- [x] SHAP explainability (both modes)
- [x] Consolidated dashboard: mode toggle + ticker tabs
- [ ] Push to GitHub
