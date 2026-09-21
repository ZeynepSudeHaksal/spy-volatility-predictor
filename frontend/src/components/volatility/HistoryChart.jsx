import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  BarChart, Bar, Cell,
} from 'recharts'

const CALM = '#2e5aac'
const TURBULENT = '#c24a1f'

function buildRows(history) {
  if (!history.dates || !history.prices || !history.regimes) return []
  return history.dates.map((date, i) => ({
    date,
    price: history.prices[i],
    regime: history.regimes[i],
  }))
}

function tickFormatter(dateStr) {
  const d = new Date(dateStr + 'T00:00:00')
  return d.toLocaleDateString('en-US', { month: 'short', year: '2-digit' })
}

function PriceTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const row = payload[0].payload
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip-date">
        {new Date(label + 'T00:00:00').toLocaleDateString('en-US', {
          month: 'short', day: 'numeric', year: 'numeric',
        })}
      </div>
      <div>${row.price.toFixed(2)}</div>
      <div className={row.regime === 'high' ? 'turbulent' : 'calm'}>
        {row.regime === 'high' ? 'High vol' : 'Low vol'}
      </div>
    </div>
  )
}

export default function HistoryChart({ history }) {
  if (!history || !history.dates || history.dates.length === 0) return null

  const rows = buildRows(history)

  return (
    <section className="section">
      <h2 className="section-title">Recent price and regime history</h2>
      <p className="section-lede">
        SPY closing price over the model's test period, with each day's
        predicted regime shown in the strip below.
      </p>

      <div className="chart-block">
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
            <XAxis
              dataKey="date"
              tickFormatter={tickFormatter}
              tick={{ fontSize: 12, fill: 'var(--color-ink-faint)' }}
              axisLine={{ stroke: 'var(--color-line)' }}
              tickLine={false}
              minTickGap={40}
            />
            <YAxis
              domain={['auto', 'auto']}
              tick={{ fontSize: 12, fill: 'var(--color-ink-faint)' }}
              axisLine={false}
              tickLine={false}
              width={44}
            />
            <Tooltip content={<PriceTooltip />} />
            <Line
              type="monotone"
              dataKey="price"
              stroke="#16181c"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={true}
            />
          </LineChart>
        </ResponsiveContainer>

        <ResponsiveContainer width="100%" height={20}>
          <BarChart data={rows} margin={{ top: 0, right: 8, bottom: 0, left: 8 }}>
            <Bar dataKey={() => 1} isAnimationActive={false}>
              {rows.map((row) => (
                <Cell
                  key={row.date}
                  fill={row.regime === 'high' ? TURBULENT : CALM}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="ribbon-legend">
          <span><span className="dot calm" /> Low-vol regime</span>
          <span><span className="dot turbulent" /> High-vol regime</span>
        </div>
      </div>
    </section>
  )
}