from scanner import scan_market
from exchange import get_balance, get_ohlcv
from strategy import liquidity_sweep

def handle_command(cmd: str):
    cmd = cmd.lower()

    # balance
    if "balance" in cmd:
        return {"type": "balance", "data": get_balance()}

    # scan market
    if "scan" in cmd:
        return {"type": "scan", "data": scan_market()[:5]}

    # btc analysis
    if "btc" in cmd:
        candles = get_ohlcv("BTC/USDT:USDT")
        signal = liquidity_sweep(candles)
        return {"type": "btc", "data": signal}

    return {
        "type": "unknown",
        "data": "Command not recognized"
    }
