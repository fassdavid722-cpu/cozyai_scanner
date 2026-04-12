#!/usr/bin/env python3
"""
Cozy Alpha Agent v3.5 — Grok Sentiment + Interactive Chat
Single‑file production‑ready trading bot with AI analyst.

Run normally: python cozy_alpha.py          (scans and trades)
Run chat mode: python cozy_alpha.py --chat  (talk to Cozy)

Environment variables expected:
- BITGET_API_KEY, BITGET_SECRET, BITGET_PASSWORD
- GROQ_API_KEY (or XAI_API_KEY)   <-- for Grok
- CHAT_ID                          <-- Telegram chat ID
- TELEGRAM_BOT_TOKEN               <-- Telegram bot token
"""

import os
import csv
import json
import pickle
import argparse
import sys
import time
from datetime import datetime, timedelta
from collections import deque

import ccxt
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

# Grok SDK (xAI)
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    print("Warning: xai-sdk not installed. Grok features disabled. Run: pip install xai-sdk")

load_dotenv()

# ======================== CONFIG ========================
BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_SECRET = os.getenv("BITGET_SECRET")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD")

# Grok API key – check both possible names
GROK_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("XAI_API_KEY")

TIMEFRAME_ENTRY = "1m"
TIMEFRAME_FILTER = "5m"
CANDLE_LIMIT = 100
MAX_SYMBOLS = 20

# Scoring weights (base, before ML adjustment)
WEIGHT_LIQUIDITY = 0.35
WEIGHT_FVG = 0.25
WEIGHT_MOMENTUM = 0.25
WEIGHT_SENTIMENT = 0.15   # Grok sentiment weight

MIN_SCORE = 0.65

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")  # <-- fixed to CHAT_ID

TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", 50))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 100))
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", 5))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 0.02))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 0.04))

ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
ML_LEARNING_RATE = float(os.getenv("ML_LEARNING_RATE", 0.01))

INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.02

# Cache for sentiment analysis (avoid hitting API too often)
SENTIMENT_CACHE = {}
CACHE_TTL_SECONDS = 900  # 15 minutes

# Identity
COZY_HANDLE = "@CozyCrypto_io"
FOUNDER_HANDLE = "@CozyCrypto_io"  # you can change this to your personal handle

# ======================== EXCHANGE ========================
exchange = ccxt.bitget({
    "apiKey": BITGET_API_KEY,
    "secret": BITGET_SECRET,
    "password": BITGET_PASSWORD,
    "options": {"defaultType": "swap"},
    "enableRateLimit": True
})

def get_symbols(limit=MAX_SYMBOLS):
    tickers = exchange.fetch_tickers()
    symbols = [s for s in tickers if ":USDT" in s]
    return symbols[:limit]

def get_ohlcv(symbol, timeframe, limit=CANDLE_LIMIT):
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

def get_current_price(symbol):
    ticker = exchange.fetch_ticker(symbol)
    return ticker["last"]

# ======================== STRATEGIES ========================
def detect_liquidity_sweep(candles):
    df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume"])
    if len(df) < 21:
        return None
    high_20 = df["high"].rolling(20).max()
    low_20 = df["low"].rolling(20).min()
    last = df.iloc[-1]
    if last["low"] < low_20.iloc[-2]:
        return {"side": "long", "score": 0.80, "reason": "Liquidity sweep below prior lows"}
    if last["high"] > high_20.iloc[-2]:
        return {"side": "short", "score": 0.80, "reason": "Liquidity sweep above prior highs"}
    return None

def detect_fvg(candles):
    df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume"])
    if len(df) < 3:
        return None
    last = df.iloc[-1]
    second_last = df.iloc[-2]
    third_last = df.iloc[-3]
    if last["low"] < second_last["high"] and last["close"] > last["open"]:
        if third_last["high"] < last["low"]:
            return {"type": "bullish", "score": 0.75, "reason": "Bullish FVG fill"}
    if last["high"] > second_last["low"] and last["close"] < last["open"]:
        if third_last["low"] > last["high"]:
            return {"type": "bearish", "score": 0.75, "reason": "Bearish FVG fill"}
    return None

def detect_momentum(candles):
    df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume"])
    if len(df) < 14:
        return None
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.iloc[-1]
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    last_vol = df["volume"].iloc[-1]
    vol_surge = last_vol > avg_vol * 2
    if last_rsi > 50 and vol_surge:
        return {"side": "long", "score": 0.70, "reason": f"Momentum bullish (RSI {last_rsi:.1f})"}
    if last_rsi < 50 and vol_surge:
        return {"side": "short", "score": 0.70, "reason": f"Momentum bearish (RSI {last_rsi:.1f})"}
    return None

def detect_sentiment(symbol):
    """Use Grok to analyze market sentiment for a symbol."""
    if not GROK_AVAILABLE or not GROK_API_KEY:
        return None

    base = symbol.split('/')[0].replace('USDT', '').replace('USD', '')
    now = datetime.utcnow().timestamp()
    if base in SENTIMENT_CACHE:
        cached = SENTIMENT_CACHE[base]
        if now - cached["timestamp"] < CACHE_TTL_SECONDS:
            return cached["data"]

    try:
        client = XAIClient(api_key=GROK_API_KEY)
        model = "grok-2-1212"
        prompt = f"""Analyze current crypto market sentiment for {base}. Consider X (Twitter) chatter, recent news, and on-chain activity. Return a JSON object with:
        - "score": float between -1.0 (extremely bearish) and 1.0 (extremely bullish)
        - "reason": short (max 120 chars) summary.
        Only valid JSON, no other text."""
        chat = client.chat.create(model=model)
        chat.append(system("You are a crypto sentiment analyst. Output only JSON."))
        chat.append(user(prompt))
        response = chat.sample()
        result = json.loads(response.content)
        normalized_score = (result["score"] + 1) / 2
        sentiment_data = {
            "side": "long" if result["score"] > 0 else "short",
            "score": normalized_score,
            "reason": f"Sentiment: {result['reason']}",
            "raw_score": result["score"]
        }
        SENTIMENT_CACHE[base] = {"timestamp": now, "data": sentiment_data}
        return sentiment_data
    except Exception as e:
        print(f"Sentiment error for {symbol}: {e}")
        return None

# ======================== MULTI‑TIMEFRAME FILTER ========================
def filter_trend(symbol):
    try:
        candles_5m = get_ohlcv(symbol, TIMEFRAME_FILTER, limit=30)
        df = pd.DataFrame(candles_5m, columns=["time","open","high","low","close","volume"])
        ema = df["close"].ewm(span=20, adjust=False).mean()
        if len(ema) < 3:
            return "neutral"
        slope = ema.iloc[-1] - ema.iloc[-3]
        if slope > 0:
            return "long"
        elif slope < 0:
            return "short"
        else:
            return "neutral"
    except Exception as e:
        print(f"5m filter error {symbol}: {e}")
        return "neutral"

# ======================== ML MEMORY ========================
class MLMemory:
    def __init__(self, model_path="ml_model.pkl", data_path="ml_data.csv"):
        self.model_path = model_path
        self.data_path = data_path
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.X_buffer = deque(maxlen=500)
        self.y_buffer = deque(maxlen=500)
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            print("ML model loaded.")
        else:
            print("No ML model found. Starting fresh.")
        if os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = pd.DataFrame(columns=[
                "timestamp", "symbol", "liq_score", "fvg_score", "mom_score", "sent_score",
                "composite_score", "side", "outcome", "pnl_pct"
            ])

    def extract_features(self, signal_data):
        comp = signal_data["components"]
        liq = comp.get("liquidity", {})
        fvg = comp.get("fvg", {})
        mom = comp.get("momentum", {})
        sent = comp.get("sentiment", {})
        return np.array([
            liq.get("score", 0),
            fvg.get("score", 0),
            mom.get("score", 0),
            sent.get("score", 0),
            signal_data["score"]
        ]).reshape(1, -1)

    def predict_weight_adjustment(self, signal_data):
        if not ML_ENABLED or not hasattr(self.model, "predict_proba"):
            return 1.0
        X = self.extract_features(signal_data)
        proba = self.model.predict_proba(X)[0]
        confidence = proba[1] if len(proba) > 1 else 0.5
        multiplier = 0.8 + 0.4 * confidence
        return multiplier

    def log_trade_outcome(self, signal_data, pnl_pct):
        comp = signal_data["components"]
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": signal_data.get("symbol", ""),
            "liq_score": comp.get("liquidity", {}).get("score", 0),
            "fvg_score": comp.get("fvg", {}).get("score", 0),
            "mom_score": comp.get("momentum", {}).get("score", 0),
            "sent_score": comp.get("sentiment", {}).get("score", 0),
            "composite_score": signal_data["score"],
            "side": signal_data["side"],
            "outcome": 1 if pnl_pct > 0 else 0,
            "pnl_pct": pnl_pct
        }
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        self.df.to_csv(self.data_path, index=False)
        X = self.extract_features(signal_data)[0]
        self.X_buffer.append(X)
        self.y_buffer.append(row["outcome"])
        if len(self.X_buffer) >= 10:
            self._retrain()

    def _retrain(self):
        X = np.array(self.X_buffer)
        y = np.array(self.y_buffer)
        self.model.fit(X, y)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        print("ML model updated.")

ml_memory = MLMemory()

# ======================== SCORING ========================
def calculate_composite_score(signals):
    total = 0
    weights_sum = 0
    sides = []
    if "liquidity" in signals:
        total += signals["liquidity"]["score"] * WEIGHT_LIQUIDITY
        weights_sum += WEIGHT_LIQUIDITY
        sides.append(signals["liquidity"]["side"])
    if "fvg" in signals:
        side = "long" if signals["fvg"]["type"] == "bullish" else "short"
        total += signals["fvg"]["score"] * WEIGHT_FVG
        weights_sum += WEIGHT_FVG
        sides.append(side)
    if "momentum" in signals:
        total += signals["momentum"]["score"] * WEIGHT_MOMENTUM
        weights_sum += WEIGHT_MOMENTUM
        sides.append(signals["momentum"]["side"])
    if "sentiment" in signals:
        total += signals["sentiment"]["score"] * WEIGHT_SENTIMENT
        weights_sum += WEIGHT_SENTIMENT
        sides.append(signals["sentiment"]["side"])
    if weights_sum == 0:
        return None, None
    base_score = total / weights_sum
    side = max(set(sides), key=sides.count) if sides else None
    return base_score, side

# ======================== SIGNAL ENGINE ========================
def evaluate_signal(symbol):
    candles_1m = get_ohlcv(symbol, TIMEFRAME_ENTRY)
    signals = {}
    liq = detect_liquidity_sweep(candles_1m)
    if liq: signals["liquidity"] = liq
    fvg = detect_fvg(candles_1m)
    if fvg: signals["fvg"] = fvg
    mom = detect_momentum(candles_1m)
    if mom: signals["momentum"] = mom
    sent = detect_sentiment(symbol)
    if sent: signals["sentiment"] = sent

    if not signals:
        return None

    base_score, side = calculate_composite_score(signals)
    if base_score is None or side is None:
        return None

    trend = filter_trend(symbol)
    if trend != "neutral" and trend != side:
        return None

    reasons = [v["reason"] for v in signals.values()]
    signal_data = {
        "side": side,
        "score": base_score,
        "reasons": reasons,
        "components": signals,
        "symbol": symbol
    }

    ml_multiplier = ml_memory.predict_weight_adjustment(signal_data)
    adjusted_score = base_score * ml_multiplier
    signal_data["score"] = adjusted_score
    signal_data["ml_multiplier"] = ml_multiplier

    if adjusted_score < MIN_SCORE:
        return None

    return signal_data

# ======================== RISK MANAGER ========================
class RiskManager:
    def __init__(self):
        self.daily_pnl = 0
        self.last_reset = datetime.utcnow().date()
        self.position_open = False
        self.current_stop_loss = None
        self.current_take_profit = None

    def can_trade(self):
        today = datetime.utcnow().date()
        if today != self.last_reset:
            self.daily_pnl = 0
            self.last_reset = today
        if self.daily_pnl <= -MAX_DAILY_LOSS:
            print(f"⚠️ Daily loss limit reached ({-MAX_DAILY_LOSS} USDT). Trading halted.")
            return False
        if self.position_open:
            print("Position already open.")
            return False
        return True

    def calculate_position_size(self, balance, entry_price, leverage=DEFAULT_LEVERAGE):
        risk_amount = balance * RISK_PER_TRADE
        return min(risk_amount * leverage, MAX_POSITION_SIZE)

    def set_stop_loss_take_profit(self, entry_price, side):
        if side == "long":
            sl = entry_price * (1 - STOP_LOSS_PERCENT)
            tp = entry_price * (1 + TAKE_PROFIT_PERCENT)
        else:
            sl = entry_price * (1 + STOP_LOSS_PERCENT)
            tp = entry_price * (1 - TAKE_PROFIT_PERCENT)
        self.current_stop_loss = sl
        self.current_take_profit = tp
        return sl, tp

risk_manager = RiskManager()

# ======================== TRADE EXECUTOR ========================
class TradeExecutor:
    def __init__(self):
        self.paper_trader = PaperTrader()

    def execute_signal(self, symbol, signal_data):
        if TRADE_MODE == "paper":
            entry_price = get_current_price(symbol)
            self.paper_trader.open_trade(symbol, signal_data["side"], entry_price)
            risk_manager.position_open = True
            return {"status": "paper_trade_opened", "entry_price": entry_price}
        elif TRADE_MODE == "live":
            if not risk_manager.can_trade():
                return {"status": "blocked_by_risk_manager"}
            try:
                exchange.set_leverage(DEFAULT_LEVERAGE, symbol)
            except Exception as e:
                print(f"Leverage set error: {e}")
            balance = exchange.fetch_balance().get("USDT", {}).get("free", 0)
            entry_price = get_current_price(symbol)
            position_size = risk_manager.calculate_position_size(balance, entry_price)
            amount = position_size / entry_price
            side = signal_data["side"]
            sl_price, tp_price = risk_manager.set_stop_loss_take_profit(entry_price, side)
            try:
                if side == "long":
                    order = exchange.create_market_buy_order(symbol, amount)
                else:
                    order = exchange.create_market_sell_order(symbol, amount)
                sl_order = exchange.create_order(
                    symbol, "limit", "sell" if side == "long" else "buy",
                    amount, sl_price, {"reduceOnly": True}
                )
                tp_order = exchange.create_order(
                    symbol, "limit", "sell" if side == "long" else "buy",
                    amount, tp_price, {"reduceOnly": True}
                )
                risk_manager.position_open = True
                print(f"✅ LIVE TRADE OPENED: {side.upper()} {symbol} @ {entry_price}")
                send_alert(symbol, signal_data, extra=f"LIVE order placed. SL: {sl_price:.4f} TP: {tp_price:.4f}")
                return {"status": "live_trade_opened", "entry_price": entry_price, "amount": amount, "sl": sl_price, "tp": tp_price}
            except Exception as e:
                print(f"Live trade error: {e}")
                send_alert(symbol, signal_data, extra=f"⚠️ LIVE TRADE FAILED: {e}")
                return {"status": "error", "error": str(e)}
        else:
            return {"status": "invalid_mode"}

executor = TradeExecutor()

# ======================== PAPER TRADER ========================
class PaperTrader:
    def __init__(self, balance=INITIAL_BALANCE, log_file="trades.csv"):
        self.balance = balance
        self.position = None
        self.log_file = log_file
        if not os.path.exists(log_file):
            with open(log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp","symbol","side","entry_price","exit_price","pnl","balance"])

    def open_trade(self, symbol, side, entry_price):
        if self.position:
            return
        self.position = {"symbol": symbol, "side": side, "entry_price": entry_price, "timestamp": datetime.utcnow()}
        print(f"📝 Paper trade opened: {side.upper()} {symbol} @ {entry_price}")
        risk_manager.position_open = True

    def close_trade(self, exit_price, signal_data=None):
        if not self.position:
            return
        entry = self.position["entry_price"]
        side = self.position["side"]
        if side == "long":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry
        pnl_usdt = (self.balance * RISK_PER_TRADE) * pnl_pct * DEFAULT_LEVERAGE
        self.balance += pnl_usdt
        risk_manager.daily_pnl += pnl_usdt
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(), self.position["symbol"], side,
                entry, exit_price, round(pnl_usdt,2), round(self.balance,2)
            ])
        print(f"📝 Paper trade closed: PnL {pnl_usdt:.2f} USDT | Balance: {self.balance:.2f}")
        if signal_data and ML_ENABLED:
            ml_memory.log_trade_outcome(signal_data, pnl_pct)
        self.position = None
        risk_manager.position_open = False
        return pnl_usdt

    def update_open_position(self, current_price):
        if not self.position:
            return
        side = self.position["side"]
        sl = risk_manager.current_stop_loss
        tp = risk_manager.current_take_profit
        if side == "long":
            if current_price <= sl or current_price >= tp:
                self.close_trade(current_price)
        else:
            if current_price >= sl or current_price <= tp:
                self.close_trade(current_price)

# ======================== TELEGRAM ========================
def send_alert(symbol, signal_data, extra=""):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    msg = f"""
🚨 *Cozy Alpha Signal*
🤖 Trading for {COZY_HANDLE}

📈 *Symbol:* `{symbol}`
🎯 *Side:* {signal_data['side'].upper()}
💯 *Score:* {signal_data['score']:.2f}
📊 *ML Boost:* {signal_data.get('ml_multiplier', 1.0):.2f}x

📋 *Reasons:*
{chr(10).join(f"• {r}" for r in signal_data['reasons'])}

{extra}
⏰ *Time:* {datetime.utcnow().isoformat()}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        print(f"Telegram error: {e}")

# ======================== COZY CHAT (Grok) ========================
class CozyChat:
    def __init__(self):
        if not GROK_AVAILABLE or not GROK_API_KEY:
            print("Grok not available. Chat mode disabled.")
            self.client = None
            return
        self.client = XAIClient(api_key=GROK_API_KEY)
        self.model = "grok-2-1212"
        self.context = []
        self.identity = f"""You are Cozy, the AI trading agent for {COZY_HANDLE}.
You were built by the founder of {COZY_HANDLE} to scan markets, detect liquidity sweeps, fair value gaps, momentum, and on-chain sentiment.
You're helpful, concise, slightly sassy, and you always remember you're part of the Cozy ecosystem.
Keep answers under 300 characters unless asked for detail."""

    def ask(self, user_input):
        if not self.client:
            return "I'm sorry, my Grok connection is offline. Check GROQ_API_KEY."
        system_msg = system(self.identity)
        self.context.append(user(user_input))
        chat = self.client.chat.create(model=self.model)
        chat.append(system_msg)
        for msg in self.context[-5:]:
            chat.append(msg)
        try:
            response = chat.sample()
            self.context.append(response)
            return response.content
        except Exception as e:
            return f"Error: {e}"

# ======================== MAIN SCANNER ========================
def run():
    print(f"🚀 Cozy Alpha v3.5 running | Mode: {TRADE_MODE} | ML: {ML_ENABLED} | Grok: {GROK_AVAILABLE and GROK_API_KEY is not None}")
    print(f"🤖 Trading for {COZY_HANDLE}")
    symbols = get_symbols()
    best_signal = None
    best_symbol = None

    for symbol in symbols:
        try:
            signal = evaluate_signal(symbol)
            if signal and (not best_signal or signal["score"] > best_signal["score"]):
                best_signal = signal
                best_symbol = symbol
        except Exception as e:
            print(f"{symbol} error: {e}")

    if best_signal:
        best_signal["timestamp"] = datetime.utcnow().isoformat()
        print(f"\n🚀 SIGNAL: {best_symbol} | {best_signal['side'].upper()} | Score: {best_signal['score']:.2f} (ML x{best_signal.get('ml_multiplier',1.0):.2f})")
        for r in best_signal["reasons"]:
            print(f"   - {r}")

        result = executor.execute_signal(best_symbol, best_signal)
        if result["status"] == "paper_trade_opened":
            send_alert(best_symbol, best_signal, extra="Paper trade opened.")
            entry = result["entry_price"]
            risk_manager.set_stop_loss_take_profit(entry, best_signal["side"])
            executor.paper_trader.position = {
                "symbol": best_symbol,
                "side": best_signal["side"],
                "entry_price": entry,
                "timestamp": datetime.utcnow()
            }
        elif result["status"] == "live_trade_opened":
            pass
    else:
        print("No signal")

    if TRADE_MODE == "paper" and risk_manager.position_open:
        pos = executor.paper_trader.position
        if pos:
            current_price = get_current_price(pos["symbol"])
            executor.paper_trader.update_open_position(current_price)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="Enter interactive chat mode with Cozy")
    args = parser.parse_args()

    if args.chat:
        cozy = CozyChat()
        print(f"💬 Cozy Chat Mode. Trading for {COZY_HANDLE}. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = cozy.ask(user_input)
            print(f"Cozy: {response}\n")
        sys.exit(0)

    while True:
        run()
        print("Waiting 60 seconds...")
        time.sleep(60)
