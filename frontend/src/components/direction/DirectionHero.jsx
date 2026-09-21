function formatDate(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr + 'T00:00:00')
  return d.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })
}

const REASON_MAP = {
  momentum_5d: 'recent price momentum',
  momentum_10d: 'price momentum over the past two weeks',
  momentum_20d: 'price momentum over the past month',
  rsi_14: 'RSI positioning',
  macd: 'the MACD trend signal',
  macd_hist: 'accelerating trend momentum',
  price_vs_ma20: 'price relative to its 20-day average',
  price_vs_ma50: 'price relative to its 50-day average',
  volume_ratio: 'above-average trading volume',
  realized_vol_20d: 'recent volatility levels',
  vix_level: 'the market-wide fear gauge',
  vix_spy_spread: 'options pricing relative to recent moves',
  log_return: "yesterday's price move",
}

function buildSummarySentence(prediction) {
  const top = prediction.top_features?.[0]
  if (!top) return null
  const reason = REASON_MAP[top.feature] || top.feature.replace(/_/g, ' ')
  return `Driven mainly by ${reason}.`
}

export default function DirectionHero({ prediction, ticker }) {
  const isUp = prediction.direction === 'up'
  const label = isUp ? 'Up expected' : 'Down expected'
  const summary = buildSummarySentence(prediction)
  const baseline = prediction.always_up_baseline

  return (
    <section className="hero">
      <p className="hero-eyebrow">{ticker}, 5-day forward outlook</p>
      <h1 className={`hero-state ${isUp ? 'up' : 'down'}`}>{label}</h1>
      <p className="hero-meta">
        <span className="hero-probability">
          {Math.round(prediction.probability * 100)}%
        </span>{' '}
        probability, as of {formatDate(prediction.date)}
      </p>
      {baseline !== null && baseline !== undefined && (
        <p className="hero-baseline">
          For comparison, this ticker has closed up {Math.round(baseline * 100)}% of
          5-day windows in the test period &mdash; that's the bar a model has to
          clear to add real value, not 50%.
        </p>
      )}
      {summary && <p className="hero-summary">{summary}</p>}
    </section>
  )
}
