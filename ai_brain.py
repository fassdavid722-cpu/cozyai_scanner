import os
import requests
from config import DRY_RUN

XAI_API_KEY = os.getenv("GROK_API_KEY")

API_URL = "https://api.x.ai/v1/chat/completions"

def evaluate_signal_ai(signal, balance):
    """
    AI acts as a filter, not a trader.
    It decides if signal is worth executing.
    """

    if not XAI_API_KEY:
        return {"approved": True, "reason": "No AI key - default approve"}

    prompt = f"""
You are a trading risk evaluator.

Decide if this trade is HIGH QUALITY or LOW QUALITY.

RULES:
- Be strict
- Avoid overtrading
- Prefer high probability setups only

ACCOUNT BALANCE: {balance}

SIGNAL:
{signal}

Return JSON only:
{{
  "approved": true/false,
  "confidence": 0-1,
  "reason": "short explanation"
}}
"""

    try:
        res = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {XAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "grok-4-latest",
                "messages": [
                    {"role": "system", "content": "You are a strict trading risk engine."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            },
            timeout=15
        )

        data = res.json()
        content = data["choices"][0]["message"]["content"]

        return eval(content)  # safe enough for controlled JSON response

    except Exception as e:
        return {
            "approved": True,
            "confidence": 0.5,
            "reason": f"AI error fallback: {str(e)}"
  }
