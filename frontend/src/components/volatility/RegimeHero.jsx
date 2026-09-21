function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr + 'T00:00:00')
  return d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
}

function buildSummarySentence(prediction) {
  const top = prediction.top_features?.[0]
  if (!top) return null

  const isHigh = prediction.regime === 'high'
  const reasonMap = {
    atr: 'wide recent price swings',
    atr_pct: 'wide recent price swings relative to price level',
    volume_ratio: 'above-average trading volume',
    vix_spy_spread: 'options pricing in more fear than recent moves justify',
    vix_level: 'an elevated VIX',
    realized_vol_5d: 'elevated volatility over the past week',
    realized_vol_10d: 'elevated volatility over the past two weeks',
    realized_vol_20d: 'elevated volatility over the past month',
    log_return: 'an unusually large move yesterday',
    abs_return: 'unusually large recent daily moves',
  }
  const reason = reasonMap[top.feature] || top.feature.replace(/_/g, ' ')

  return isHigh
    ? `Flagged mainly by ${reason}.`
    : `Recent conditions look calm, driven mainly by the absence of ${reason}.`
}

export default function RegimeHero({ prediction, ticker }) {
  const isHigh = prediction.regime === 'high'
  const label = isHigh ? 'High volatility expected' : 'Low volatility expected'
  const summary = buildSummarySentence(prediction)

  return (
    <section className="hero">
      <p className="hero-eyebrow">{ticker}, 5-day forward outlook</p>
      <h1 className={`hero-state ${isHigh ? 'turbulent' : 'calm'}`}>{label}</h1>
      <p className="hero-meta">
        <span className="hero-probability">
          {Math.round(prediction.probability * 100)}%
        </span>{' '}
        probability, as of {formatDate(prediction.date)}
      </p>
      {summary && <p className="hero-summary">{summary}</p>}
    </section>
  )
}
