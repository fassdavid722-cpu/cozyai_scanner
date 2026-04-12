import ccxt
from config import BITGET_API_KEY, BITGET_SECRET, BITGET_PASSWORD

exchange = ccxt.bitget({
    "apiKey": BITGET_API_KEY,
    "secret": BITGET_SECRET,
    "password": BITGET_PASSWORD,
    "options": {"defaultType": "swap"},
    "enableRateLimit": True,
})

def get_symbols(limit=30):
    tickers = exchange.fetch_tickers()
    symbols = [s for s in tickers if s.endswith(":USDT")]
    return symbols[:limit]

def get_ohlcv(symbol, timeframe="1m", limit=100):
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def get_balance():
    bal = exchange.fetch_balance()
    return bal["total"].get("USDT", 0)
