// Plain-language translations for each engineered feature, written from
// the perspective of a PM/analyst reading the dashboard -- not the
// column name itself. This is the whole point of surfacing SHAP values
// at all: "why" needs to be legible to someone who isn't going to look
// up what ATR stands for.
const FEATURE_COPY = {
  atr: 'Recent daily price swings have been wide',
  atr_pct: 'Price swings, relative to the current price level',
  volume_ratio: 'Trading volume is running above its recent average',
  vix_spy_spread: 'Options pricing suggests more fear than recent price action alone would justify',
  vix_level: 'The market\u2019s overall fear gauge (VIX) is elevated',
  realized_vol_5d: 'Volatility over the last week has been elevated',
  realized_vol_10d: 'Volatility over the last two weeks has been elevated',
  realized_vol_20d: 'Volatility over the last month has been elevated',
  log_return: 'Yesterday\u2019s move was unusually large',
  abs_return: 'The size of recent daily moves has been large',
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
        Each factor below either pushed the prediction toward a high-volatility
        read or pulled it back toward calm, ranked by how much it mattered.
      </p>
      <ul className="feature-list">
        {topFeatures.map((f) => {
          const pushesHigh = f.shap_value > 0
          const widthPct = (Math.abs(f.shap_value) / maxAbs) * 100
          return (
            <li className="feature-row" key={f.feature}>
              <div className="feature-text">
                <span className="feature-desc">{describeFeature(f.feature)}</span>
                <span
                  className={`feature-direction ${pushesHigh ? 'turbulent' : 'calm'}`}
                >
                  {pushesHigh ? 'toward high vol' : 'toward calm'}
                </span>
              </div>
              <div className="feature-bar-track">
                <div
                  className={`feature-bar-fill ${pushesHigh ? 'turbulent' : 'calm'}`}
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
