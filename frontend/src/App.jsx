import { useEffect, useState } from 'react'
import { fetchTickers, fetchLatestPrediction, fetchHistory } from './api'
import ModeToggle from './components/ModeToggle'
import TickerSelector from './components/TickerSelector'
import RegimeHero from './components/volatility/RegimeHero'
import VolFeatureBreakdown from './components/volatility/FeatureBreakdown'
import VolHistoryChart from './components/volatility/HistoryChart'
import DirectionHero from './components/direction/DirectionHero'
import DirFeatureBreakdown from './components/direction/FeatureBreakdown'
import DirHistoryChart from './components/direction/HistoryChart'

const MODE_META = {
  volatility: {
    footer: `Model: XGBoost, explained with SHAP. One pooled model trained
      across SPY, QQQ, IWM, DIA, XLF, GLD, and TLT, predicting each ticker's own
      5-day forward realized volatility regime relative to its own
      recent history. Evaluated on a chronologically held-out test
      period. A risk-flagging tool, not a trading signal \u2014 for
      portfolio triage and research use.`,
  },
  direction: {
    footer: `Model: XGBoost, explained with SHAP. Predicts 5-day forward price
      direction (up/down) for the same basket. Direction is a
      fundamentally harder, noisier problem than volatility regime \u2014
      results are always shown against the "always predict up" baseline,
      which reflects real historical drift, not a 50% coin flip. A
      research tool, not investment advice.`,
  },
}

function App() {
  const [mode, setMode] = useState('volatility')
  const [tickers, setTickers] = useState([])
  const [selectedTicker, setSelectedTicker] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [history, setHistory] = useState(null)
  const [error, setError] = useState(null)

  // Reload the ticker list whenever the mode changes (each mode tracks
  // its own available tickers and default independently).
  useEffect(() => {
    setPrediction(null)
    setHistory(null)
    fetchTickers(mode)
      .then((data) => {
        setTickers(data.tickers)
        setSelectedTicker(data.default || data.tickers[0])
      })
      .catch(() => setError('Could not reach the prediction API.'))
  }, [mode])

  useEffect(() => {
    if (!selectedTicker) return
    setPrediction(null)
    setHistory(null)
    Promise.all([
      fetchLatestPrediction(mode, selectedTicker),
      fetchHistory(mode, selectedTicker),
    ])
      .then(([pred, hist]) => {
        setPrediction(pred)
        setHistory(hist)
      })
      .catch(() => setError('Could not reach the prediction API.'))
  }, [mode, selectedTicker])

  const notReady =
    prediction && (mode === 'volatility' ? prediction.regime === null : prediction.direction === null)

  return (
    <div className="page">
      <header className="masthead">
        <p className="masthead-label">Market Analytics</p>
        <ModeToggle mode={mode} onSelect={setMode} />
        <TickerSelector tickers={tickers} selected={selectedTicker} onSelect={setSelectedTicker} />
      </header>

      <main className="content">
        {error && <p className="status-message">{error}</p>}
        {!error && !prediction && <p className="status-message">Loading&hellip;</p>}
        {notReady && (
          <p className="status-message">
            {prediction.note || 'Run the model pipeline to see a live prediction.'}
          </p>
        )}

        {prediction && !notReady && mode === 'volatility' && (
          <>
            <RegimeHero prediction={prediction} ticker={selectedTicker} />
            <div className="divider" />
            <VolFeatureBreakdown topFeatures={prediction.top_features} />
            <div className="divider" />
            <VolHistoryChart history={history} />
          </>
        )}

        {prediction && !notReady && mode === 'direction' && (
          <>
            <DirectionHero prediction={prediction} ticker={selectedTicker} />
            <div className="divider" />
            <DirFeatureBreakdown topFeatures={prediction.top_features} />
            <div className="divider" />
            <DirHistoryChart history={history} />
          </>
        )}
      </main>

      <footer className="footer">
        <p>{MODE_META[mode].footer}</p>
      </footer>
    </div>
  )
}

export default App
