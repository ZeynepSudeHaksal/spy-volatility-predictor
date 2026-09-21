// Plain-language descriptions of each feature, written for a PM/analyst
// reading the dashboard. The direction each one pushed (toward up or
// down) comes from the SHAP value's sign, shown separately -- these
// descriptions are neutral statements of what the indicator measures.
const FEATURE_COPY = {
  momentum_5d: 'Price momentum over the past week',
  momentum_10d: 'Price momentum over the past two weeks',
  momentum_20d: 'Price momentum over the past month',
  rsi_14: 'RSI (14-day) \u2014 overbought/oversold positioning',
  macd: 'MACD trend indicator',
  macd_hist: 'MACD trend acceleration (histogram)',
  price_vs_ma20: 'Price relative to its 20-day average',
  price_vs_ma50: 'Price relative to its 50-day average',
  volume_ratio: 'Trading volume relative to its recent average',
  realized_vol_20d: 'Realized volatility over the past month',
  vix_level: 'Market-wide fear gauge (VIX)',
  vix_spy_spread: 'VIX relative to recent realized volatility',
  log_return: "Yesterday's price move",
}

function describeFeature(featureKey) {
  return FEATURE_COPY[featureKey] || featureKey.replace(/_/g, ' ')
}

export default function FeatureBreakdown({ topFeatures }) {
  if (!topFeatures || topFeatures.length === 0) return null

  const maxAbs = Math.max(...topFeatures.map((f) => Math.abs(f.shap_value)))

  return (
    <section className="section">
      <h2 className="section-title">Why the model thinks this</h2>
      <p className="section-lede">
        Each factor below either pushed the prediction toward "up" or
        pulled it back toward "down," ranked by how much it mattered.
      </p>
      <ul className="feature-list">
        {topFeatures.map((f) => {
          const pushesUp = f.shap_value > 0
          const widthPct = (Math.abs(f.shap_value) / maxAbs) * 100
          return (
            <li className="feature-row" key={f.feature}>
              <div className="feature-text">
                <span className="feature-desc">{describeFeature(f.feature)}</span>
                <span className={`feature-direction ${pushesUp ? 'up' : 'down'}`}>
                  {pushesUp ? 'toward up' : 'toward down'}
                </span>
              </div>
              <div className="feature-bar-track">
                <div
                  className={`feature-bar-fill ${pushesUp ? 'up' : 'down'}`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
