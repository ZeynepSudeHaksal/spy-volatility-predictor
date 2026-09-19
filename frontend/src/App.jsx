import { useEffect, useState } from 'react'
import { fetchLatestPrediction, fetchHistory } from './api'
import RegimeHero from './components/RegimeHero'
import FeatureBreakdown from './components/FeatureBreakdown'
import HistoryChart from './components/HistoryChart'

function App() {
  const [prediction, setPrediction] = useState(null)
  const [history, setHistory] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([fetchLatestPrediction(), fetchHistory()])
      .then(([pred, hist]) => {
        setPrediction(pred)
        setHistory(hist)
      })
      .catch(() => setError('Could not reach the prediction API.'))
  }, [])

  const notReady = prediction && prediction.regime === null

  return (
    <div className="page">
      <header className="masthead">
        <p className="masthead-label">SPY Volatility Regime Predictor</p>
      </header>

      <main className="content">
        {error && <p className="status-message">{error}</p>}

        {!error && !prediction && <p className="status-message">Loading&hellip;</p>}

        {notReady && (
          <p className="status-message">
            {prediction.note || 'Run the model pipeline to see a live prediction.'}
          </p>
        )}

        {prediction && !notReady && (
          <>
            <RegimeHero prediction={prediction} />
            <div className="divider" />
            <FeatureBreakdown topFeatures={prediction.top_features} />
            <div className="divider" />
            <HistoryChart history={history} />
          </>
        )}
      </main>

      <footer className="footer">
        <p>
          Model: XGBoost, explained with SHAP. Predicts 5-day forward realized
          volatility regime for SPY, evaluated on a chronologically held-out
          test period. A risk-flagging tool, not a trading signal &mdash;
          for portfolio triage and research use.
        </p>
      </footer>
    </div>
  )
}

export default App
