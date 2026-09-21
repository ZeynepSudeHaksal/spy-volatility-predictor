const API_BASE = 'http://localhost:5001/api'

export async function fetchTickers(mode) {
  const res = await fetch(`${API_BASE}/${mode}/tickers`)
  if (!res.ok) throw new Error('Failed to fetch tickers')
  return res.json()
}

export async function fetchLatestPrediction(mode, ticker) {
  const res = await fetch(`${API_BASE}/${mode}/predict/latest?ticker=${ticker}`)
  if (!res.ok) throw new Error('Failed to fetch latest prediction')
  return res.json()
}

export async function fetchHistory(mode, ticker) {
  const res = await fetch(`${API_BASE}/${mode}/history?ticker=${ticker}`)
  if (!res.ok) throw new Error('Failed to fetch history')
  return res.json()
}
