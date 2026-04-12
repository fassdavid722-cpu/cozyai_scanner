import pandas as pd

def liquidity_sweep(candles):
    df = pd.DataFrame(candles, columns=["t","o","h","l","c","v"])
    if len(df) < 25:
        return None

    high = df["h"].rolling(20).max()
    low = df["l"].rolling(20).min()
    last = df.iloc[-1]

    if last["l"] < low.iloc[-2]:
        return {"side": "long", "score": 0.8, "reason": "sell-side sweep"}

    if last["h"] > high.iloc[-2]:
        return {"side": "short", "score": 0.8, "reason": "buy-side sweep"}

    return None


def volume_confirmation(candles):
    df = pd.DataFrame(candles, columns=["t","o","h","l","c","v"])
    avg = df["v"].rolling(20).mean().iloc[-1]
    last = df["v"].iloc[-1]

    if avg == 0:
        return 1.0

    return min(2.0, last / avg)
