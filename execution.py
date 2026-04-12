best = signals[0]

print("Best signal:", best)

if best["score"] < MIN_SCORE:
    time.sleep(COOLDOWN_SECONDS)
    continue

# 🧠 AI FILTER LAYER
ai_decision = evaluate_signal_ai(best, balance)

print("AI Decision:", ai_decision)

if not ai_decision.get("approved"):
    print("AI rejected trade")
    time.sleep(COOLDOWN_SECONDS)
    continue

risk_amount = calculate_risk(balance, best["score"])

result = place_trade(
    best["symbol"],
    best["side"],
    risk_amount
)

print("TRADE RESULT:", result)
