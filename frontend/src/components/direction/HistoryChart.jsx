import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip,
  BarChart, Bar, Cell,
} from 'recharts'

const DOWN = '#c24a1f'
const UP = '#2e7d4f'

function buildRows(history) {
  if (!history.dates || !history.prices || !history.directions) return []
  return history.dates.map((date, i) => ({
    date,
    price: history.prices[i],
    direction: history.directions[i],
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
      <div className={row.direction === 'up' ? 'up' : 'down'}>
        {row.direction === 'up' ? 'Predicted up' : 'Predicted down'}
      </div>
    </div>
  )
}

export default function HistoryChart({ history }) {
  if (!history || !history.dates || history.dates.length === 0) return null

  const rows = buildRows(history)

  return (
    <section className="section">
      <h2 className="section-title">Recent price and prediction history</h2>
      <p className="section-lede">
        Closing price over the model's test period, with each day's predicted
        5-day-forward direction shown in the strip below.
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
            <Line type="monotone" dataKey="price" stroke="#16181c" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>

        <ResponsiveContainer width="100%" height={20}>
          <BarChart data={rows} margin={{ top: 0, right: 8, bottom: 0, left: 8 }}>
            <Bar dataKey={() => 1} isAnimationActive={false}>
              {rows.map((row) => (
                <Cell key={row.date} fill={row.direction === 'up' ? UP : DOWN} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div className="ribbon-legend">
          <span><span className="dot up" /> Predicted up</span>
          <span><span className="dot down" /> Predicted down</span>
        </div>
      </div>
    </section>
  )
}