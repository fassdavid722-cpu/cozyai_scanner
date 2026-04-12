import time
from ai_brain import evaluate_signal_ai
from scanner import scan_market
from exchange import get_balance
from risk import calculate_risk
from execution import place_trade
from config import MIN_SCORE, COOLDOWN_SECONDS, MAX_TRADES_PER_CYCLE

last_trade_time = 0

def run():
    global last_trade_time

    print("Cozy TraderAgent v1 started...")

    while True:
        try:
            balance = get_balance()
            signals = scan_market()

            if not signals:
                print("No signals")
                time.sleep(COOLDOWN_SECONDS)
                continue

            best = signals[0]

            print("Best signal:", best)

            if best["score"] < MIN_SCORE:
                time.sleep(COOLDOWN_SECONDS)
                continue

            if time.time() - last_trade_time < COOLDOWN_SECONDS:
                continue

            risk_amount = calculate_risk(balance, best["score"])

            result = place_trade(
                best["symbol"],
                best["side"],
                risk_amount
            )

            print("TRADE RESULT:", result)

            last_trade_time = time.time()

        except Exception as e:
            print("Error:", e)

        time.sleep(COOLDOWN_SECONDS)


if __name__ == "__main__":
    run()
