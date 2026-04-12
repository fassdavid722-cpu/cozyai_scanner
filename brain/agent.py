from tools.market import scan_symbols, get_price
from tools.risk import calculate_risk

def think(balance):
    symbols = scan_symbols()
    best = None

    for symbol in symbols:
        price = get_price(symbol)

        if best is None:
            best = {"symbol": symbol, "price": price}

    risk = calculate_risk(balance)

    return {
        "trade_symbol": best["symbol"],
        "risk": risk
    }
