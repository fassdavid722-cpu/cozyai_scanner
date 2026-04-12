from config import DRY_RUN
import ccxt
from exchange import exchange

def place_trade(symbol, side, usdt_amount):
    if DRY_RUN:
        return {
            "status": "DRY_RUN",
            "symbol": symbol,
            "side": side,
            "amount": usdt_amount
        }

    price = exchange.fetch_ticker(symbol)["last"]
    size = usdt_amount / price

    order = exchange.create_market_order(symbol, side, size)

    return order
