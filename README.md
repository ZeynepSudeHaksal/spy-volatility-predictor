# SPY Volatility Regime Predictor

An explainable ML classifier that predicts whether SPY's realized volatility
over the next 5 trading days will be **elevated** or **normal**, relative to
its own recent history. Built as a risk-flagging tool for a
portfolio-manager-facing audience, not a trading signal.

## Why volatility regime, not price direction

Next-day/next-week price *direction* is close to a coin flip for a liquid
index like SPY (efficient market hypothesis holds up reasonably well at
that horizon). Volatility, on the other hand, is well documented to
**cluster** -- calm periods tend to stay calm, turbulent periods tend to
stay turbulent -- which makes it a more tractable and more honest target
for a beginner-to-intermediate ML project.

## Target definition

Classify whether **forward 5-day realized volatility** will exceed its own
**trailing 60-day median** (binary: high-vol regime vs. low-vol regime).
Using a rolling, relative threshold (rather than a fixed cutoff like
"vol > 20%") keeps the label meaningful across a multi-year dataset
spanning very different volatility environments.

## Pipeline

| Script | Purpose |
|---|---|
| `01_fetch_data.py` | Pull raw SPY OHLCV + VIX data via yfinance |
| `02_engineer_features.py` | Realized vol, ATR, volume anomalies, VIX spread, etc. |
| `03_label_data.py` | Compute forward-vol labels |
| `04_train_test_split.py` | Walk-forward (time-respecting) split |
| `05_train_model.py` | Logistic Regression baseline -> XGBoost |
| `06_evaluate.py` | Metrics + SHAP explainability |
| `backend/app.py` | Flask API serving predictions to the frontend |
| `frontend/` | React (Vite) dashboard |

## Project layout

```
spy-volatility-predictor/
├── scripts/          # 01-06 pipeline (data -> features -> labels -> model)
├── data/{raw,processed}/
├── notebooks/         # exploratory work
├── backend/           # Flask API
│   ├── app.py
│   └── requirements.txt
└── frontend/           # React (Vite) dashboard
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── index.css
```

## Setup -- ML pipeline

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Setup -- backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py   # runs on http://localhost:5001
```

## Setup -- frontend

```bash
cd frontend
npm install
npm run dev   # runs on http://localhost:5173
```

## Usage

```bash
python scripts/01_fetch_data.py
```

Fetches SPY and VIX daily data (2005-present by default) and saves to
`data/raw/`. Override the date range with `--start` / `--end`.

With the backend and frontend both running, open `http://localhost:5173`
to see the current placeholder dashboard -- it confirms the frontend can
reach the Flask API. It'll return empty/null data until the model
pipeline (scripts 02-06) is built out.

## Roadmap

- [x] Data pull (SPY + VIX)
- [x] Backend + frontend scaffold (Flask + React, wired together)
- [ ] Feature engineering
- [ ] Labeling
- [ ] Walk-forward split
- [ ] Model training + SHAP
- [ ] Full dashboard design (charts, regime history, SHAP explanations)
- [ ] Extend to a basket of tickers
