#!/usr/bin/env python3
"""
Cozy Alpha Agent v3.5 — Grok Sentiment + Telegram Chat Listener
Single‑file trading bot with always‑on chat via Telegram.

Environment variables (set in GitHub Secrets or .env):
- BITGET_API_KEY, BITGET_SECRET, BITGET_PASSWORD
- XAI_API_KEY (Grok)
- CHAT_ID
- TELEGRAM_BOT_TOKEN
- TRADE_MODE (paper or live)
"""

import os
import csv
import json
import pickle
import argparse
import sys
import time
import threading
from datetime import datetime
from collections import deque

import ccxt
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression

# Grok SDK (optional, graceful fallback)
try:
    from xai_sdk import Client as XAIClient
    from xai_sdk.chat import user, system
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    print("Warning: xai-sdk not installed. Grok features disabled.")

load_dotenv()

# ======================== CONFIG ========================
BITGET_API_KEY = os.getenv("BITGET_API_KEY")
BITGET_SECRET = os.getenv("BITGET_SECRET")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD")

GROK_API_KEY = os.getenv("XAI_API_KEY")  # Grok API key

TIMEFRAME_ENTRY = "1m"
TIMEFRAME_FILTER = "5m"
CANDLE_LIMIT = 100
MAX_SYMBOLS = 20

WEIGHT_LIQUIDITY = 0.35
WEIGHT_FVG = 0.25
WEIGHT_MOMENTUM = 0.25
WEIGHT_SENTIMENT = 0.15

MIN_SCORE = 0.65

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("CHAT_ID")

TRADE_MODE = os.getenv("TRADE_MODE", "paper").lower()
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", 50))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", 100))
DEFAULT_LEVERAGE = int(os.getenv("DEFAULT_LEVERAGE", 5))
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", 0.02))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", 0.04))

ML_ENABLED = os.getenv("ML_ENABLED", "true").lower() == "true"
INITIAL_BALANCE = 1000
RISK_PER_TRADE = 0.02

SENTIMENT_CACHE = {}
CACHE_TTL_SECONDS = 900

COZY_HANDLE = "@CozyCrypto_io"

# Telegram polling offset
last_update_id = 0

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
        prompt = f"""Analyze current crypto market sentiment for {base}. Return JSON:
        {{"score": float between -1.0 and 1.0, "reason": "short summary max 120 chars"}}"""
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
        print(f"Sentiment error {symbol}: {e}")
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
        return 0.8 + 0.4 * confidence

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
            return False
        if self.position_open:
            return False
        return True

    def calculate_position_size(self, balance, entry_price):
        risk_amount = balance * RISK_PER_TRADE
        return min(risk_amount * DEFAULT_LEVERAGE, MAX_POSITION_SIZE)

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
        self.position = {"symbol": symbol, "side": side, "entry_price": entry_price}
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
        # Live trading omitted for safety; add later if needed.

executor = TradeExecutor()

# ======================== TELEGRAM & CHAT ========================
class CozyChat:
    def __init__(self):
        self.client = None
        if GROK_AVAILABLE and GROK_API_KEY:
            self.client = XAIClient(api_key=GROK_API_KEY)
        self.model = "grok-2-1212"
        self.identity = f"""You are Cozy, AI trading agent for {COZY_HANDLE}.
You use liquidity sweeps, FVGs, momentum, and sentiment.
Keep answers under 300 chars unless asked for detail. Be helpful and slightly sassy."""

    def ask(self, user_input):
        if not self.client:
            return "Grok connection offline. Check XAI_API_KEY."
        try:
            chat = self.client.chat.create(model=self.model)
            chat.append(system(self.identity))
            chat.append(user(user_input))
            response = chat.sample()
            return response.content
        except Exception as e:
            return f"Error: {e}"

cozy_chat = CozyChat()

def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=5)
    except Exception as e:
        print(f"Telegram send error: {e}")

def process_telegram_command(command, args):
    cmd = command.lower()
    if cmd == "/ask":
        question = " ".join(args)
        if not question:
            return "Ask me something! Example: /ask What's BTC doing?"
        return cozy_chat.ask(question)
    elif cmd == "/status":
        pos = executor.paper_trader.position
        if pos:
            return f"📊 Open {pos['side'].upper()} on {pos['symbol']} @ {pos['entry_price']:.4f}"
        else:
            return "No open position. Scanning for setups."
    elif cmd == "/balance":
        bal = executor.paper_trader.balance
        return f"💰 Balance: {bal:.2f} USDT"
    elif cmd == "/help":
        return "Commands:\n/ask <question>\n/status\n/balance\n/help"
    else:
        return "Unknown command. Try /help"

def telegram_polling():
    global last_update_id
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    while True:
        try:
            params = {"offset": last_update_id + 1, "timeout": 30}
            resp = requests.get(url, params=params, timeout=35).json()
            if resp.get("ok"):
                for update in resp["result"]:
                    last_update_id = update["update_id"]
                    msg = update.get("message", {})
                    text = msg.get("text", "")
                    chat_id = msg.get("chat", {}).get("id")
                    if text.startswith("/"):
                        parts = text.split()
                        cmd = parts[0]
                        args = parts[1:]
                        reply = process_telegram_command(cmd, args)
                        reply_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                        requests.post(reply_url, json={"chat_id": chat_id, "text": reply}, timeout=5)
        except Exception as e:
            print(f"Telegram polling error: {e}")
        time.sleep(1)

# ======================== MAIN SCANNER ========================
def run():
    print(f"🚀 Cozy Alpha v3.5 | Mode: {TRADE_MODE} | ML: {ML_ENABLED} | Grok: {GROK_AVAILABLE}")
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
        print(f"\n🚀 SIGNAL: {best_symbol} | {best_signal['side'].upper()} | Score: {best_signal['score']:.2f}")
        for r in best_signal["reasons"]:
            print(f"   - {r}")
        result = executor.execute_signal(best_symbol, best_signal)
        if result["status"] == "paper_trade_opened":
            send_telegram_message(f"🚨 New Signal: {best_symbol} {best_signal['side'].upper()} (Score: {best_signal['score']:.2f})")
            entry = result["entry_price"]
            risk_manager.set_stop_loss_take_profit(entry, best_signal["side"])
    else:
        print("No signal")

    if TRADE_MODE == "paper" and risk_manager.position_open:
        pos = executor.paper_trader.position
        if pos:
            current_price = get_current_price(pos["symbol"])
            executor.paper_trader.update_open_position(current_price)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument("--chat", action="store_true", help="CLI chat mode")
    args = parser.parse_args()

    if args.chat:
        print(f"💬 CLI Chat Mode. Trading for {COZY_HANDLE}. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            print(f"Cozy: {cozy_chat.ask(user_input)}\n")
        sys.exit(0)

    if args.once:
        run()
        sys.exit(0)

    # Start Telegram polling in background thread
    if TELEGRAM_BOT_TOKEN:
        telegram_thread = threading.Thread(target=telegram_polling, daemon=True)
        telegram_thread.start()
        print("📱 Telegram listener started. Send /help to your bot.")

    # Main scanner loop
    while True:
        run()
        print("Waiting 60 seconds...")
        time.sleep(60)
