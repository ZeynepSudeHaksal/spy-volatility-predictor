# Market Analytics (Volatility Regime + Direction Prediction)

Two ML prediction modes for a basket of index/sector ETFs (SPY, QQQ, IWM,
DIA, XLF), sharing one data pipeline foundation, one Flask backend, and
one React dashboard with a top-level mode toggle:

- **Volatility regime** (the stronger result -- see Findings below):
  will 5-day forward realized volatility be elevated or normal,
  relative to each ticker's own recent history? A risk-flagging tool
  for portfolio triage.
- **Direction**: will price be up or down 5 trading days out? A
  deliberately harder, more honestly-framed problem, and one where the
  rigorous result (see Findings) is a legitimate negative -- see below
  for why that's a feature of this project's methodology, not a bug.

Both are pooled models (one XGBoost model per mode, `ticker` as a
one-hot feature, trained across the whole basket), explained with SHAP,
evaluated with a per-ticker purge gap to prevent label lookahead
leakage.

## Findings

**Volatility regime**: ROC-AUC 0.59-0.66 across all five tickers (see
`models/volatility/metadata.json` for exact numbers) -- consistently,
clearly above random, and the project's strongest signal. But raw
accuracy (45-56% depending on ticker) actually trails the naive
"always predict normal" baseline (~58-65%), and that's by design, not
a bug: both models use class weighting
(`LogisticRegression(class_weight="balanced")` / XGBoost's
`scale_pos_weight`) to deliberately prioritize *recall* on the
high-volatility class -- for a risk-flagging tool, missing a real
high-vol day (a false negative) is worse than an unnecessary alert
(a false positive), so trading some accuracy for much better recall
(65-78% across tickers) is the right tradeoff, not a shortcoming.

I tried to have it both ways -- `scripts/volatility/common.py`'s
`select_f1_threshold()` picks a decision threshold on the validation
set to get a better balance than the arbitrary 0.5 default. An earlier
version optimized the threshold for raw accuracy directly, which
"worked" (beat the baseline) by pushing the threshold so high recall
collapsed to ~6% overall and 0% on one ticker (XLF) -- accuracy went
up by making the model nearly useless at its actual job. F1 avoids
that collapse, but confirms there's no threshold that gives good
accuracy *and* good recall here: that's the real finding, and ROC-AUC
(threshold-independent) is the fairer metric for judging this model.

**Direction**: this is where the project's methodology mattered most.
A single train/test split gave inconsistent, sometimes alarming
results (ROC-AUC swinging from ~0.51 down to ~0.45 depending on minor
pipeline changes) -- not because anything was broken, but because a
single split of a few hundred rows per ticker doesn't have enough
statistical power to reliably detect an effect this small. Running a
proper rolling walk-forward backtest (`07_walk_forward_backtest.py`,
34 folds, ~16,500 aggregated out-of-fold predictions spanning
2010-2027) resolved this with real statistical power:

- **XGBoost**: ROC-AUC ~0.51 -- indistinguishable from random guessing.
- **Logistic Regression**: ROC-AUC ~0.51 -- also indistinguishable
  from random on the most recent re-run. An earlier run (less data)
  showed LR at 0.528, which read as a small but statistically real
  edge at the time -- re-running with several more months of data
  pulled it back down to the noise band around 0.50. That shift is
  itself informative: it means the original "edge" was sitting close
  enough to the noise floor that it didn't hold up, which is exactly
  the kind of thing a walk-forward re-run is supposed to catch before
  you trust a borderline result.
- **Neither model beats the "always predict up" accuracy baseline**
  (~60%). ROC-AUC and raw accuracy answer different questions: ROC-AUC
  asks "can it rank up-days above down-days better than chance" (no,
  for either model now); beating a 60% baseline in raw accuracy needs
  a much bigger edge than beating a 50% coin flip does. Both things
  point the same direction here.

**Bottom line for direction**: no real, usable signal from either
model at a 5-day horizon -- a legitimate negative result for a
problem that published research also finds hard (liquid index ETFs
are close to efficiently priced at this timescale). That's the
honest, rigorously-supported conclusion, not a failure of the
pipeline. The walk-forward backtest exists specifically because a
naive single-split evaluation could not have told "broken pipeline"
and "genuinely small/nonexistent effect" apart -- and, as the LR
number above shows, a single walk-forward run on less data can still
read as a real effect that more data later shows isn't there.

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
│   ├── cv_utils.py               # SHARED: purged walk-forward CV split generator
│   ├── volatility/                # 02-06 + optional 05a tuning, + common.py
│   └── direction/                  # 02-06 + optional 05a tuning, + common.py
├── data/
│   ├── raw/                      # shared
│   └── processed/{volatility,direction}/
├── models/{volatility,direction}/  # + tuned_hyperparameters.json if 05a was run
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
python scripts/volatility/05a_tune_hyperparameters.py   # optional -- see below
python scripts/volatility/05_train_model.py               # REQUIRED before direction/02b below
python scripts/volatility/06_evaluate.py

python scripts/direction/02_engineer_features.py
python scripts/direction/02b_add_volatility_regime.py    # optional -- adds the trained
                                                            # volatility model's regime
                                                            # probability as a direction feature.
                                                            # Requires volatility/05_train_model.py
                                                            # to have already run.
python scripts/direction/03_label_data.py                 # now volatility-adjusted, with an
                                                            # ambiguous-move threshold -- see
                                                            # --threshold-z below
python scripts/direction/04_train_test_split.py
python scripts/direction/05a_tune_hyperparameters.py      # optional -- see below
python scripts/direction/05_train_model.py
python scripts/direction/06_evaluate.py
```

### Direction labeling: volatility-adjusted, with an ambiguous-move threshold

`direction/03_label_data.py` no longer labels every day as strictly
up/down. A 1% move means something different on a calm ticker than a
turbulent one, so the forward 5-day return is first converted to a
z-score against that ticker's own typical 5-day move (derived from
`realized_vol_20d`), and rows where that z-score falls inside
`--threshold-z` (default 0.25) of zero are dropped entirely -- from
both training and evaluation -- rather than forced into an arbitrary
up/down call on moves that were genuinely noise. Adjust with:

```bash
python scripts/direction/03_label_data.py --threshold-z 0.35
```

### Direction feature: volatility regime probability (optional)

`direction/02b_add_volatility_regime.py` scores every row with the
already-trained volatility model and adds its predicted regime
probability as a direction feature -- testing whether the volatility
model's refined signal carries information beyond direction's own raw
technical features. Read that script's docstring for an important
caveat on how to interpret this feature: the volatility model's
predictions are somewhat optimistic on dates that were in its own
training set, so weight direction's TEST-period results here more
heavily than train-period results.

### Ablation: does a feature actually help?

`direction/05_train_model.py --exclude-features <name1,name2,...>`
drops the named feature(s) before training, so you can directly compare
model performance with vs. without any feature -- most usefully
`vol_regime_probability`, to check whether it's actually earning its
place rather than just adding noise:

```bash
python scripts/direction/05_train_model.py --exclude-features vol_regime_probability
python scripts/direction/06_evaluate.py
# compare against a normal run (no --exclude-features) and see which
# has the higher TEST ROC-AUC
```

**A word of caution before drawing conclusions from any of these
comparisons**: with a test set in the hundreds of rows per ticker,
ROC-AUC has real sampling noise -- roughly `sqrt(0.25/n)`, so ~500 rows
gives a noise band around +/-0.02, ~100 rows more like +/-0.05. A
swing that size between two runs isn't necessarily a real difference.
Don't over-read a single comparison; look for a consistent pattern
across tickers and, ideally, across a few different train/test date
windows before concluding a feature genuinely helps or hurts.

### Rolling walk-forward backtest (recommended over the single split)

`direction/04_train_test_split.py` through `06_evaluate.py` test ONE
thing: does a model trained once on old data still work on newer data?
If it doesn't, that could mean the approach doesn't work at all -- or
it could mean the market regime shifted and a model frozen at one
point in time just can't keep up, which is a solvable problem
(retrain periodically), not a dead end.

`direction/07_walk_forward_backtest.py` tests the more realistic
scenario directly: it retrains fresh on a rolling trailing window
(default 5 years) every few months (default a 6-month step), walks
that forward through the FULL available history, and collects
genuinely out-of-sample predictions the whole way -- typically dozens
of folds and thousands of out-of-fold predictions, a far more
statistically robust estimate than one single train/test split. It
also prints per-fold results over time, so a regime-shift pattern
(some periods harder than others) shows up directly rather than being
guessed at from one split's result.

This script also trains and deploys the actual PRODUCTION model --
fit on the most recent trailing window, which is what a real
deployment would use, replacing the single frozen model
`05_train_model.py` would have produced. It writes the same backend
artifacts `06_evaluate.py` does (so the dashboard needs no changes),
except the "history" shown is now every out-of-fold walk-forward
prediction across all of history -- a more honest historical record
than a single test split, since every point in it was genuinely
out-of-sample at the time.

```bash
python scripts/direction/07_walk_forward_backtest.py
python scripts/direction/07_walk_forward_backtest.py --window-years 5 --step-months 6
python scripts/direction/07_walk_forward_backtest.py --exclude-features vol_regime_probability
```

Requires `labeled_data.csv` (run `02_engineer_features.py` through
`03_label_data.py` first -- `04_train_test_split.py` is NOT needed,
since this script walks the full history itself rather than using a
single static split).

### Optional: hyperparameter tuning (05a)

Each pipeline's `05_train_model.py` uses sensible hand-picked defaults out
of the box -- `05a_tune_hyperparameters.py` is optional but recommended
before a "final" run. It searches for better XGBoost and Logistic
Regression hyperparameters using **purged walk-forward cross-validation**
on the train split only (never val or test), via `scripts/cv_utils.py`.
Standard random k-fold CV would leak information across time (the same
reasons `04_train_test_split.py`'s walk-forward split exists in the first
place); this avoids that by expanding the training window fold by fold
and purging each ticker's last few training rows before every validation
window. Run it, then run `05_train_model.py` as usual -- it automatically
picks up `models/<mode>/tuned_hyperparameters.json` if present, and falls
back to the defaults if not:

```bash
python scripts/volatility/05a_tune_hyperparameters.py --n-splits 4 --n-iter 25
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

## Running it with Docker

The training pipeline (`scripts/`) still runs locally with Python -- it's a
one-time (or periodic) batch job, not something that needs a container.
What's containerized is the two pieces that actually need to *run as a
service*: the Flask API and the static frontend.

```bash
docker compose up --build
```

- Dashboard: http://localhost:8080
- API health check: http://localhost:5001/api/health

Two separate images, defined in `backend/Dockerfile` and
`frontend/Dockerfile`:

- **`backend/Dockerfile`** -- `app.py` only reads the pre-computed
  `backend/data/*/*.json` files the pipeline already wrote and serves them;
  it never imports pandas/sklearn/xgboost/shap at runtime. So the image
  installs just `flask` + `gunicorn` (`backend/requirements.txt` is now
  serving-only deps -- the training stack stays in the repo-root
  `requirements.txt`), runs as a non-root user, and serves via gunicorn
  instead of Flask's single-threaded dev server. `FRONTEND_ORIGIN` (CORS)
  and `PORT` are both overridable env vars rather than hardcoded, so the
  same image works locally and in a real deployment.
- **`frontend/Dockerfile`** -- multi-stage: builds the Vite app in a
  throwaway `node` image, ships only the static output in `nginx:alpine`.
  `VITE_API_BASE` is a build arg (Vite bakes env vars in at build time, not
  runtime) -- set it to wherever the API actually ends up living.

To run the backend image alone, without compose:

```bash
cd backend
docker build -t market-analytics-backend .
docker run -p 5001:5001 -e FRONTEND_ORIGIN=http://localhost:5173 market-analytics-backend
```

## Roadmap

- [x] Shared data pipeline + two independent prediction pipelines
- [x] Walk-forward split, per-ticker purge gaps, leakage guards (both modes)
- [x] SHAP explainability (both modes)
- [x] Consolidated dashboard: mode toggle + ticker tabs
- [x] Hyperparameter tuning via purged walk-forward CV (both modes)
- [x] Direction: volatility-adjusted labeling, cross-pipeline feature (vol
      regime as a direction input), full rolling walk-forward backtest --
      see Findings above for the result
- [x] Push to GitHub
- [x] Containerize backend (Flask/gunicorn) + frontend (static build/nginx),
      one-command local deploy via `docker compose up --build`
- [ ] Deploy to a public URL
- [ ] CI (GitHub Actions): lint/tests on push, build the Docker images
- [ ] Optional: cross-sectional/relative features for direction, if
      revisited (see Findings -- the small real signal found there is
      the only remaining lead worth pursuing for improving accuracy)