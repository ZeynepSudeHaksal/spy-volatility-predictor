import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip,
  BarChart, Bar, Cell, CartesianGrid, Brush,
} from 'recharts'
import DateRangeControl, { useDateRange } from '../shared/DateRangeControl'

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
      <div className="chart-tooltip-value">${row.price.toFixed(2)}</div>
      <div className={row.direction === 'up' ? 'up' : 'down'}>
        {row.direction === 'up' ? 'Predicted up' : 'Predicted down'}
      </div>
    </div>
  )
}

export default function HistoryChart({ history }) {
  if (!history || !history.dates || history.dates.length === 0) return null

  const allRows = buildRows(history)
  const range = useDateRange(allRows)
  const rows = range.filtered
  const lastRow = rows[rows.length - 1]

  return (
    <section className="section">
      <h2 className="section-title">Recent price and prediction history</h2>
      <p className="section-lede">
        Closing price with each day's walk-forward, out-of-sample predicted
        5-day-forward direction shown in the strip below &mdash; every point
        here is a prediction the model made without having seen that day's
        outcome. Drag the handles beneath the chart to zoom into a specific
        stretch of dates.
      </p>

      <DateRangeControl range={range} />

      <div className="chart-block">
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 8 }}>
            <defs>
              <linearGradient id="dirPriceFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#16181c" stopOpacity={0.1} />
                <stop offset="100%" stopColor="#16181c" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              vertical={false}
              stroke="var(--color-line)"
              strokeWidth={1}
            />
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
            <Tooltip
              content={<PriceTooltip />}
              cursor={{ stroke: 'var(--color-ink-faint)', strokeWidth: 1 }}
            />
            <Area
              type="monotone"
              dataKey="price"
              stroke="#16181c"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, stroke: 'var(--color-bg)', strokeWidth: 2 }}
              fill="url(#dirPriceFill)"
              isAnimationActive={true}
            />
            {rows.length > 40 && (
              <Brush
                dataKey="date"
                height={24}
                travellerWidth={8}
                tickFormatter={tickFormatter}
                stroke="var(--color-ink-faint)"
                fill="var(--color-bg)"
              />
            )}
          </AreaChart>
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
          {lastRow && (
            <span className="ribbon-legend-latest">
              Latest: ${lastRow.price.toFixed(2)} on{' '}
              {new Date(lastRow.date + 'T00:00:00').toLocaleDateString('en-US', {
                month: 'short', day: 'numeric', year: 'numeric',
              })}
            </span>
          )}
        </div>
      </div>
    </section>
  )
}