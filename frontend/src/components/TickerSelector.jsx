export default function TickerSelector({ tickers, selected, onSelect }) {
  if (!tickers || tickers.length <= 1) return null

  return (
    <nav className="ticker-nav">
      {tickers.map((ticker) => (
        <button
          key={ticker}
          className={`ticker-tab ${ticker === selected ? 'active' : ''}`}
          onClick={() => onSelect(ticker)}
        >
          {ticker}
        </button>
      ))}
    </nav>
  )
}
