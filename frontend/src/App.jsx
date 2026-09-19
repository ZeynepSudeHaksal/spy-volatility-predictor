import { useEffect, useState } from 'react'
import { fetchTickers, fetchLatestPrediction, fetchHistory } from './api'
import TickerSelector from './components/TickerSelector'
import RegimeHero from './components/RegimeHero'
import FeatureBreakdown from './components/FeatureBreakdown'
import HistoryChart from './components/HistoryChart'

function App() {
  const [tickers, setTickers] = useState([])
  const [selectedTicker, setSelectedTicker] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [history, setHistory] = useState(null)
  const [error, setError] = useState(null)

  // Load the available ticker list once, pick the default.
  useEffect(() => {
    fetchTickers()
      .then((data) => {
        setTickers(data.tickers)
        setSelectedTicker(data.default || data.tickers[0])
      })
      .catch(() => setError('Could not reach the prediction API.'))
  }, [])

  // Re-fetch prediction + history whenever the selected ticker changes.
  useEffect(() => {
    if (!selectedTicker) return
    setPrediction(null)
    setHistory(null)
    Promise.all([
      fetchLatestPrediction(selectedTicker),
      fetchHistory(selectedTicker),
    ])
      .then(([pred, hist]) => {
        setPrediction(pred)
        setHistory(hist)
      })
      .catch(() => setError('Could not reach the prediction API.'))
  }, [selectedTicker])

  const notReady = prediction && prediction.regime === null

  return (
    <div className="page">
      <header className="masthead">
        <p className="masthead-label">Volatility Regime Predictor</p>
        <TickerSelector
          tickers={tickers}
          selected={selectedTicker}
          onSelect={setSelectedTicker}
        />
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
            <RegimeHero prediction={prediction} ticker={selectedTicker} />
            <div className="divider" />
            <FeatureBreakdown topFeatures={prediction.top_features} />
            <div className="divider" />
            <HistoryChart history={history} />
          </>
        )}
      </main>

      <footer className="footer">
        <p>
          Model: XGBoost, explained with SHAP. One pooled model trained
          across SPY, QQQ, IWM, DIA, and XLF, predicting each ticker's own
          5-day forward realized volatility regime relative to its own
          recent history. Evaluated on a chronologically held-out test
          period. A risk-flagging tool, not a trading signal &mdash; for
          portfolio triage and research use.
        </p>
      </footer>
    </div>
  )
}

export default App