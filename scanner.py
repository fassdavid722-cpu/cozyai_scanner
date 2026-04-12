from exchange import get_symbols, get_ohlcv
from strategy import liquidity_sweep, volume_confirmation

def scan_market(limit=30):
    symbols = get_symbols(limit)

    signals = []

    for sym in symbols:
        try:
            candles = get_ohlcv(sym)

            sweep = liquidity_sweep(candles)
            if not sweep:
                continue

            vol = volume_confirmation(candles)

            score = sweep["score"] * vol

            signals.append({
                "symbol": sym,
                "side": sweep["side"],
                "score": score,
                "reason": sweep["reason"]
            })

        except Exception:
            continue

    return sorted(signals, key=lambda x: x["score"], reverse=True)
