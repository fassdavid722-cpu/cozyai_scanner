import ccxt

exchange = ccxt.bitget({
    "apiKey": "",
    "secret": "",
    "password": "",
    "options": {"defaultType": "swap"},
    "enableRateLimit": True
})

def scan_symbols():
    tickers = exchange.fetch_tickers()
    usdt_pairs = [s for s in tickers if ":USDT" in s]
    return usdt_pairs[:50]

def get_price(symbol):
    ticker = exchange.fetch_ticker(symbol)
    return ticker["last"]
