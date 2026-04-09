#!/usr/bin/env python3 """ CozyHybridAI — Bitget Futures Execution Engine

Includes:

Ensemble signal generation

Adaptive strategy weights

Toxic-hour learning

Symbol bias learning

One-trade-at-a-time state

Live entry with attached stop-loss / take-profit params

Position monitoring and equity reconciliation

Bootstrap sizing for tiny accounts

DRY_RUN / LIVE_TRADING guardrails


Default safety posture:

DRY_RUN=true

LIVE_TRADING=false


Before live use:

Validate in DRY_RUN

Then use Bitget demo / test mode if available

Then enable live trading with the smallest size you can safely tolerate """


import os import json import time from datetime import datetime, timedelta from typing import Dict, List, Tuple, Optional

import ccxt import numpy as np import pandas as pd import requests

======================================================

ENV / CONFIG

======================================================

BITGET_API_KEY = os.getenv("BITGET_API_KEY", "") BITGET_SECRET = os.getenv("BITGET_SECRET", "") BITGET_PASSWORD = os.getenv("BITGET_PASSWORD", "") TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "") TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true" LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true" USE_DEMO_TRADING = os.getenv("USE_DEMO_TRADING", "false").lower() == "true"

STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", "3.0")) MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05")) MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3")) MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "10.0")) MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "0.5")) RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.02")) BOOTSTRAP_THRESHOLD = float(os.getenv("BOOTSTRAP_THRESHOLD", "10.0")) BASE_THRESHOLD = float(os.getenv("BASE_THRESHOLD", "0.68")) MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "30")) TIMEFRAME = os.getenv("TIMEFRAME", "1m") OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "100")) HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "200")) TAKE_PROFIT_R = float(os.getenv("TAKE_PROFIT_R", "2.0")) MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.65"))

STRATEGY_LIST = [ "volume_breakout", "orderbook_imbalance", "funding_arbitrage", "liquidity_sweep", "momentum", ] DEFAULT_STRATEGY_WEIGHTS = {s: 1.0 / len(STRATEGY_LIST) for s in STRATEGY_LIST} MEMORY_FILE = os.getenv("COZY_MEMORY_FILE", "cozy_memory.json")

======================================================

HELPERS

======================================================

def clamp(x: float, lo: float, hi: float) -> float: return float(max(lo, min(hi, x)))

def safe_sigmoid(x: float) -> float: x = float(np.clip(x, -20.0, 20.0)) return float(1.0 / (1.0 + np.exp(-x)))

def send_telegram(message: str) -> None: print(message) if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return

try:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "HTML",
        },
        timeout=10,
    )
except Exception as exc:
    print(f"Telegram error: {exc}")

def build_exchange() -> ccxt.bitget: exchange = ccxt.bitget( { "apiKey": BITGET_API_KEY, "secret": BITGET_SECRET, "password": BITGET_PASSWORD, "options": {"defaultType": "swap"}, "enableRateLimit": True, "timeout": 20000, } )

if USE_DEMO_TRADING:
    try:
        exchange.enable_demo_trading(True)
    except Exception:
        pass

# If your Bitget/CCXT build supports sandbox mode, this is a second safety rail.
try:
    if DRY_RUN:
        exchange.set_sandbox_mode(True)
except Exception:
    pass

return exchange

def safe_call(func, retries: int = 3, delay: float = 1.5): last_exc = None for attempt in range(retries): try: return func() except Exception as exc: last_exc = exc time.sleep(delay) if last_exc: raise last_exc return None

======================================================

MEMORY

======================================================

class Memory: def init(self, file: str = MEMORY_FILE): self.file = file self.data = self.load() self.normalize()

def load(self) -> Dict:
    if os.path.exists(self.file):
        try:
            with open(self.file, "r") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "equity": STARTING_EQUITY,
        "daily_loss": 0.0,
        "last_trade_day": None,
        "consecutive_losses": 0,
        "sl_count": 0,
        "last_sl_time": None,
        "learned_biases": {},
        "toxic_hours": [],
        "performance_history": [],
        "strategy_performance": {
            s: {"wins": 0, "losses": 0, "total_r": 0.0} for s in STRATEGY_LIST
        },
        "strategy_weights": DEFAULT_STRATEGY_WEIGHTS.copy(),
        "last_trade_symbol": None,
        "last_trade_side": None,
        "open_trade": None,
    }

def normalize(self) -> None:
    self.data.setdefault("equity", STARTING_EQUITY)
    self.data.setdefault("daily_loss", 0.0)
    self.data.setdefault("last_trade_day", None)
    self.data.setdefault("consecutive_losses", 0)
    self.data.setdefault("sl_count", 0)
    self.data.setdefault("last_sl_time", None)
    self.data.setdefault("learned_biases", {})
    self.data.setdefault("toxic_hours", [])
    self.data.setdefault("performance_history", [])
    self.data.setdefault("strategy_performance", {
        s: {"wins": 0, "losses": 0, "total_r": 0.0} for s in STRATEGY_LIST
    })
    self.data.setdefault("strategy_weights", DEFAULT_STRATEGY_WEIGHTS.copy())
    self.data.setdefault("last_trade_symbol", None)
    self.data.setdefault("last_trade_side", None)
    self.data.setdefault("open_trade", None)

def save(self) -> None:
    with open(self.file, "w") as f:
        json.dump(self.data, f, indent=4)

def sync_equity_from_exchange(self, exchange) -> None:
    try:
        balance = safe_call(lambda: exchange.fetch_balance())
        total = (balance or {}).get("total", {}) or {}
        equity = float(total.get("USDT", self.data.get("equity", STARTING_EQUITY)))
        self.data["equity"] = max(equity, 0.0)
        self.save()
    except Exception:
        pass

def get_equity(self) -> float:
    return float(self.data.get("equity", STARTING_EQUITY))

def get_daily_loss(self) -> float:
    return float(self.data.get("daily_loss", 0.0))

def get_consecutive_losses(self) -> int:
    return int(self.data.get("consecutive_losses", 0))

def reset_daily_loss(self) -> None:
    self.data["daily_loss"] = 0.0
    self.save()

def record_daily_loss(self, loss_pct: float) -> None:
    self.data["daily_loss"] = float(self.data.get("daily_loss", 0.0)) + float(loss_pct)
    self.save()

def reset_consecutive_losses(self) -> None:
    self.data["consecutive_losses"] = 0
    self.save()

def record_loss(self, symbol: str) -> None:
    self.data["sl_count"] = int(self.data.get("sl_count", 0)) + 1
    self.data["last_sl_time"] = datetime.now().isoformat()
    self.data["learned_biases"][symbol] = clamp(
        self.data["learned_biases"].get(symbol, 0.0) + 0.05,
        0.0,
        0.25,
    )
    self.save()

def record_trade(
    self,
    symbol: str,
    side: str,
    score: float,
    lev: float,
    pnl_pct: float,
    strategy_used: str,
    regime: str,
    hour: int,
) -> None:
    trade = {
        "time": datetime.now().isoformat(),
        "hour": int(hour),
        "symbol": symbol,
        "side": side,
        "score": float(score),
        "lev": float(lev),
        "pnl_pct": float(pnl_pct),
        "strategy": strategy_used,
        "regime": regime,
    }

    self.data["performance_history"].append(trade)
    if len(self.data["performance_history"]) > HISTORY_LIMIT:
        self.data["performance_history"] = self.data["performance_history"][-HISTORY_LIMIT:]

    if strategy_used in self.data["strategy_performance"]:
        perf = self.data["strategy_performance"][strategy_used]
        if pnl_pct > 0:
            perf["wins"] += 1
        elif pnl_pct < 0:
            perf["losses"] += 1
        perf["total_r"] += float(pnl_pct) / 100.0

    if pnl_pct < 0:
        self.data["consecutive_losses"] = int(self.data.get("consecutive_losses", 0)) + 1
    else:
        self.data["consecutive_losses"] = 0

    self.apply_learning()
    self.save()

def apply_learning(self) -> None:
    history = self.data.get("performance_history", [])
    if len(history) < 10:
        return

    df = pd.DataFrame(history)

    # Hours where average performance is poor become toxic.
    hour_perf = df.groupby("hour")["pnl_pct"].mean()
    self.data["toxic_hours"] = [int(h) for h in hour_perf[hour_perf < -0.5].index.tolist()]

    # Penalize symbols with bad mean return.
    sym_perf = df.groupby("symbol")["pnl_pct"].mean()
    for sym, avg in sym_perf.items():
        if avg < -1.0:
            self.data["learned_biases"][sym] = clamp(
                self.data["learned_biases"].get(sym, 0.0) + 0.05,
                0.0,
                0.25,
            )

    # Decay bias slowly so the model can forgive market noise.
    for sym in list(self.data["learned_biases"].keys()):
        self.data["learned_biases"][sym] = max(0.0, self.data["learned_biases"][sym] - 0.02)
        if self.data["learned_biases"][sym] == 0:
            del self.data["learned_biases"][sym]

    # Strategy weight adaptation based on recent actual outcomes.
    recent = history[-20:]
    counts = {s: {"wins": 0, "losses": 0} for s in STRATEGY_LIST}
    for trade in recent:
        s = trade.get("strategy")
        if s in counts:
            if trade.get("pnl_pct", 0) > 0:
                counts[s]["wins"] += 1
            elif trade.get("pnl_pct", 0) < 0:
                counts[s]["losses"] += 1

    raw_weights = {}
    for s in STRATEGY_LIST:
        total = counts[s]["wins"] + counts[s]["losses"]
        if total > 0:
            win_rate = counts[s]["wins"] / total
            raw_weights[s] = max(0.05, float(win_rate))
        else:
            raw_weights[s] = DEFAULT_STRATEGY_WEIGHTS[s]

    # Smooth the update so weights don't jump violently.
    current = self.data.get("strategy_weights", DEFAULT_STRATEGY_WEIGHTS.copy())
    blended = {}
    alpha = 0.80
    for s in STRATEGY_LIST:
        blended[s] = alpha * float(current.get(s, DEFAULT_STRATEGY_WEIGHTS[s])) + (1 - alpha) * float(raw_weights[s])

    total_w = sum(blended.values()) or 1.0
    self.data["strategy_weights"] = {s: float(v / total_w) for s, v in blended.items()}

def check_reset_window(self) -> None:
    now = datetime.now()
    last_sl = self.data.get("last_sl_time")
    if last_sl:
        try:
            last_time = datetime.fromisoformat(last_sl)
            if now - last_time > timedelta(hours=24):
                self.data["sl_count"] = 0
        except Exception:
            pass

    today = now.date().isoformat()
    if self.data.get("last_trade_day") != today:
        self.data["daily_loss"] = 0.0
        self.data["last_trade_day"] = today

    self.save()

======================================================

MARKET DATA

======================================================

def fetch_ohlcv(exchange, symbol: str, timeframe: str = TIMEFRAME, limit: int = OHLCV_LIMIT) -> Optional[pd.DataFrame]: try: candles = safe_call(lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)) if not candles: return None df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"]) df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") return df except Exception as exc: print(f"OHLCV error {symbol}: {exc}") return None

def fetch_order_book_imbalance(exchange, symbol: str) -> float: try: book = safe_call(lambda: exchange.fetch_order_book(symbol, limit=10)) if not book: return 1.0 bid_vol = sum(level[1] for level in (book.get("bids") or [])[:5]) ask_vol = sum(level[1] for level in (book.get("asks") or [])[:5]) if ask_vol <= 0: return 1.0 return float(bid_vol / ask_vol) except Exception: return 1.0

def fetch_funding_rate(exchange, symbol: str) -> float: try: funding = safe_call(lambda: exchange.fetch_funding_rate(symbol)) return float((funding or {}).get("fundingRate", 0.0)) except Exception: return 0.0

def get_top_symbols(exchange, limit: int = MAX_SYMBOLS) -> List[str]: tickers = safe_call(lambda: exchange.fetch_tickers()) or {} df = pd.DataFrame.from_dict(tickers, orient="index") if df.empty or "symbol" not in df.columns: return []

df = df[df["symbol"].fillna("").str.contains(":USDT")]
if "quoteVolume" in df.columns:
    df = df.nlargest(limit, "quoteVolume")
return df.index.tolist()[:limit]

======================================================

STRATEGIES

======================================================

def compute_volume_breakout(df: pd.DataFrame) -> Tuple[float, str]: if len(df) < 20: return 0.5, "neutral"

data = df.copy()
data["range"] = (data["high"] - data["low"]).replace(0, 1e-9)
data["v_intensity"] = data["volume"] / data["range"]

mean = data["v_intensity"].mean()
std = data["v_intensity"].std() + 1e-9
z = (data["v_intensity"].iloc[-1] - mean) / std
prob = safe_sigmoid(z)

if prob > 0.70:
    return float(prob), "long"
if prob < 0.30:
    return float(1.0 - prob), "short"
return 0.5, "neutral"

def compute_orderbook_imbalance(imbalance: float) -> Tuple[float, str]: if imbalance > 1.5: return min(0.9, (imbalance - 1.5) / 2.0 + 0.6), "long" if imbalance < 0.67: return min(0.9, (0.67 - imbalance) / 0.5 + 0.6), "short" return 0.5, "neutral"

def compute_funding_arbitrage(funding_rate: float) -> Tuple[float, str]: if funding_rate < -0.001: return 0.85, "long" if funding_rate > 0.001: return 0.85, "short" return 0.5, "neutral"

def compute_liquidity_sweep(df: pd.DataFrame) -> Tuple[float, str]: if len(df) < 30: return 0.5, "neutral"

recent = df.tail(30).copy()
recent["high_20"] = recent["high"].rolling(20).max()
recent["low_20"] = recent["low"].rolling(20).min()
recent["sweep_high"] = (recent["high"] > recent["high_20"].shift(1)) & (recent["close"] < recent["high"])
recent["sweep_low"] = (recent["low"] < recent["low_20"].shift(1)) & (recent["close"] > recent["low"])

sweep = recent[recent["sweep_high"] | recent["sweep_low"]]
if sweep.empty:
    return 0.5, "neutral"

last = sweep.iloc[-1]
if bool(last["sweep_low"]):
    return 0.70, "long"
return 0.70, "short"

def compute_momentum(df: pd.DataFrame) -> Tuple[float, str]: if len(df) < 26: return 0.5, "neutral"

close = df["close"].copy()
delta = close.diff()
up = delta.clip(lower=0)
down = -delta.clip(upper=0)
avg_up = up.rolling(14).mean().iloc[-1]
avg_down = down.rolling(14).mean().iloc[-1]
rsi = 100 if avg_down == 0 else 100 - (100 / (1 + (avg_up / avg_down)))

exp12 = close.ewm(span=12, adjust=False).mean()
exp26 = close.ewm(span=26, adjust=False).mean()
macd = exp12 - exp26
signal = macd.ewm(span=9, adjust=False).mean()
hist = macd - signal

if rsi > 70 and hist.iloc[-1] < 0:
    return 0.70, "short"
if rsi < 30 and hist.iloc[-1] > 0:
    return 0.70, "long"
return 0.5, "neutral"

def detect_regime(df: pd.DataFrame) -> str: if len(df) < 30: return "ranging"

high = df["high"]
low = df["low"]
close = df["close"]

plus_dm = high.diff()
minus_dm = low.diff()
plus_dm[plus_dm < 0] = 0
minus_dm[minus_dm > 0] = 0

tr = pd.concat(
    [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
    axis=1,
).max(axis=1)
atr = tr.rolling(14).mean()

plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-9))
minus_di = 100 * (minus_dm.abs().rolling(14).mean() / (atr + 1e-9))
dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
adx = dx.rolling(14).mean().iloc[-1]

close_series = close.tail(20)
net_move = abs(close_series.iloc[-1] - close_series.iloc[0])
total_move = close_series.diff().abs().sum()
efficiency = net_move / total_move if total_move != 0 else 0.0

if adx > 25 and efficiency > 0.3:
    return "trending"
return "ranging"

def estimate_stop_distance_pct(df: pd.DataFrame) -> float: prev_close = df["close"].shift(1) tr = pd.concat( [ df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs(), ], axis=1, ).max(axis=1) atr = tr.rolling(14).mean().iloc[-1] price = float(df["close"].iloc[-1]) return max((atr * 1.2) / price, 0.006)

======================================================

SIZING

======================================================

def compute_position_size(equity: float, confidence: float, recent_trades: List[Dict]) -> float: confidence = clamp(confidence, 0.0, 1.0) risk_pct = RISK_PER_TRADE_PCT * confidence

if len(recent_trades) >= 10:
    wins = [t["pnl_pct"] for t in recent_trades if t.get("pnl_pct", 0) > 0]
    losses = [abs(t["pnl_pct"]) for t in recent_trades if t.get("pnl_pct", 0) < 0]
    win_rate = len(wins) / max(1, len(recent_trades))
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 1.0
    if avg_win > 0 and avg_loss > 0:
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = clamp(kelly, 0.0, 0.10)
        risk_pct = min(risk_pct * (1.0 + kelly), 0.10)

risk_amount = equity * risk_pct
return max(MIN_NOTIONAL, float(risk_amount))

======================================================

MAIN ENGINE

======================================================

class CozyHybridAI: def init(self): self.memory = Memory() self.exchange = build_exchange() try: self.exchange.load_markets() except Exception: pass

def _update_trade_day(self) -> None:
    today = datetime.now().date().isoformat()
    if self.memory.data.get("last_trade_day") != today:
        self.memory.reset_daily_loss()
        self.memory.data["last_trade_day"] = today
        self.memory.save()

def _survival_gate(self) -> bool:
    self.memory.check_reset_window()
    if self.memory.get_consecutive_losses() >= MAX_CONSECUTIVE_LOSSES:
        send_telegram(f"🛑 HALTED: {self.memory.get_consecutive_losses()} consecutive losses")
        return False
    if self.memory.get_daily_loss() >= MAX_DAILY_LOSS_PCT:
        send_telegram(f"⛔ HALTED: daily loss {self.memory.get_daily_loss() * 100:.1f}%")
        return False
    return True

def _position_from_exchange(self, symbol: str) -> Optional[Dict]:
    try:
        positions = safe_call(lambda: self.exchange.fetch_positions([symbol])) or []
        for p in positions:
            if p.get("symbol") != symbol:
                continue
            contracts = p.get("contracts")
            if contracts is None:
                contracts = p.get("size") or p.get("positionAmt") or 0
            try:
                contracts = abs(float(contracts))
            except Exception:
                contracts = 0.0
            if contracts > 0:
                return p
    except Exception:
        return None
    return None

def _is_flat(self, symbol: str) -> bool:
    return self._position_from_exchange(symbol) is None

def _finalize_closed_trade(self, open_trade: Dict) -> None:
    symbol = open_trade["symbol"]
    entry_equity = float(open_trade.get("entry_equity", self.memory.get_equity()))
    prev_eq = self.memory.get_equity()
    self.memory.sync_equity_from_exchange(self.exchange)
    new_eq = self.memory.get_equity()

    realized = new_eq - prev_eq
    pnl_pct = 0.0 if entry_equity <= 0 else (realized / entry_equity) * 100.0

    if realized < 0:
        self.memory.record_loss(symbol)
    else:
        self.memory.reset_consecutive_losses()

    self.memory.record_trade(
        symbol=symbol,
        side=open_trade.get("side", "long"),
        score=float(open_trade.get("score", 0.0)),
        lev=float(open_trade.get("lev", 1.0)),
        pnl_pct=float(pnl_pct),
        strategy_used=open_trade.get("dominant_strategy", "ensemble"),
        regime=open_trade.get("regime", "unknown"),
        hour=int(open_trade.get("hour", datetime.now().hour)),
    )

    if realized < 0:
        self.memory.record_daily_loss(abs(pnl_pct) / 100.0)

    self.memory.data["open_trade"] = None
    self.memory.data["last_trade_symbol"] = symbol
    self.memory.data["last_trade_side"] = open_trade.get("side")
    self.memory.save()

    send_telegram(
        f"📘 Trade closed\n"
        f"Symbol: {symbol}\n"
        f"PnL: {pnl_pct:.2f}%\n"
        f"Equity: ${new_eq:.2f}"
    )

def _manage_open_trade(self) -> None:
    open_trade = self.memory.data.get("open_trade")
    if not open_trade:
        return

    symbol = open_trade.get("symbol")
    if not symbol:
        self.memory.data["open_trade"] = None
        self.memory.save()
        return

    pos = self._position_from_exchange(symbol)
    if pos is not None:
        # Position still open. We intentionally keep management simple and safe.
        # Entry order should already have attached stop loss / take profit.
        return

    # No open position found; reconcile as a closed trade.
    self._finalize_closed_trade(open_trade)

def _combine_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    imbalance = fetch_order_book_imbalance(self.exchange, symbol)
    funding = fetch_funding_rate(self.exchange, symbol)
    regime = detect_regime(df)

    strategies: List[Tuple[float, str, str]] = []
    s1, d1 = compute_volume_breakout(df)
    strategies.append((s1, d1, "volume_breakout"))

    s2, d2 = compute_orderbook_imbalance(imbalance)
    strategies.append((s2, d2, "orderbook_imbalance"))

    s3, d3 = compute_funding_arbitrage(funding)
    strategies.append((s3, d3, "funding_arbitrage"))

    s4, d4 = compute_liquidity_sweep(df)
    strategies.append((s4, d4, "liquidity_sweep"))

    if regime == "trending":
        s5, d5 = compute_momentum(df)
        strategies.append((s5, d5, "momentum"))

    weights = self.memory.data.get("strategy_weights", DEFAULT_STRATEGY_WEIGHTS.copy())
    total_weight = 0.0
    weighted_score = 0.0
    direction_votes = {"long": 0.0, "short": 0.0, "neutral": 0.0}

    for score, direction, strat in strategies:
        w = float(weights.get(strat, 0.1))
        total_weight += w
        weighted_score += score * w
        direction_votes[direction] += w

    if total_weight <= 0:
        return None

    combined_score = weighted_score / total_weight
    if direction_votes["long"] > direction_votes["short"]:
        direction = "long"
    elif direction_votes["short"] > direction_votes["long"]:
        direction = "short"
    else:
        direction = "neutral"

    hour = datetime.now().hour
    bias = float(self.memory.data.get("learned_biases", {}).get(symbol, 0.0))
    penalty = 1.0 - bias
    if hour in self.memory.data.get("toxic_hours", []):
        penalty -= 0.08

    final_score = clamp(combined_score * max(0.1, penalty), 0.0, 1.0)
    dominant_strategy = max(
        strategies,
        key=lambda x: x[0] * float(weights.get(x[2], 0.1)),
    )[2] if strategies else "ensemble"

    return {
        "symbol": symbol,
        "direction": direction,
        "regime": regime,
        "raw_score": float(combined_score),
        "final_score": float(final_score),
        "bias": float(bias),
        "hour": int(hour),
        "dominant_strategy": dominant_strategy,
    }

def _calculate_trade_params(self, stop_dist_pct: float) -> Dict:
    equity = self.memory.get_equity()
    stop_dist_pct = max(float(stop_dist_pct), 0.006)

    if equity < BOOTSTRAP_THRESHOLD:
        mode = "BOOTSTRAP"
        size = MIN_NOTIONAL
        lev = max(1.0, size / max(equity, 0.01))
    else:
        mode = "PROFESSIONAL"
        ideal_size = (equity * RISK_PER_TRADE_PCT) / stop_dist_pct
        size = max(MIN_NOTIONAL, ideal_size)
        lev = max(1.0, size / max(equity, 0.01))

    lev = clamp(lev, 1.0, MAX_LEVERAGE)
    return {"mode": mode, "size": round(size, 4), "lev": round(lev, 2)}

def _place_live_entry(self, symbol: str, side: str, amount: float, sl_price: float, tp_price: float):
    order_side = "buy" if side == "long" else "sell"
    params = {
        "stopLossPrice": sl_price,
        "takeProfitPrice": tp_price,
    }
    # CCXT documents attached stop-loss / take-profit params for create_order.
    return safe_call(lambda: self.exchange.create_order(symbol, "market", order_side, amount, None, params=params))

def _attempt_set_leverage(self, symbol: str, leverage: int) -> None:
    try:
        self.exchange.set_margin_mode("isolated", symbol)
    except Exception:
        pass
    try:
        self.exchange.set_leverage(int(leverage), symbol)
    except Exception:
        pass

def _enter_trade(self, signal: Dict, df: pd.DataFrame) -> None:
    symbol = signal["symbol"]
    side = signal["direction"]
    price = float(df["close"].iloc[-1])
    stop_dist_pct = estimate_stop_distance_pct(df)
    params = self._calculate_trade_params(stop_dist_pct)

    # Respect the $3 account bootstrap + exchange minimum.
    amount_usdt = params["size"]
    amount_base = amount_usdt / price
    try:
        amount_base = float(self.exchange.amount_to_precision(symbol, amount_base))
    except Exception:
        amount_base = float(amount_base)

    if amount_base <= 0:
        send_telegram(f"⚠️ Amount precision invalid for {symbol}")
        return

    if side == "long":
        sl_price = price * (1.0 - stop_dist_pct)
        tp_price = price * (1.0 + stop_dist_pct * TAKE_PROFIT_R)
    else:
        sl_price = price * (1.0 + stop_dist_pct)
        tp_price = price * (1.0 - stop_dist_pct * TAKE_PROFIT_R)

    msg = (
        f"🚀 <b>HYBRID AI SIGNAL</b>\n"
        f"Symbol: {symbol}\n"
        f"Direction: {side.upper()}\n"
        f"Confidence: {signal['final_score']:.2f}\n"
        f"Raw Score: {signal['raw_score']:.2f}\n"
        f"Mode: {params['mode']}\n"
        f"Position: ${amount_usdt:.2f}\n"
        f"Leverage: {params['lev']:.2f}x\n"
        f"Regime: {signal['regime']}\n"
        f"Dominant Strategy: {signal['dominant_strategy']}\n"
        f"Bias Tax: {signal['bias']:.2f}\n"
        f"Price: {price:.6f}\n"
        f"SL: {sl_price:.6f}\n"
        f"TP: {tp_price:.6f}"
    )
    send_telegram(msg)

    if DRY_RUN or not LIVE_TRADING:
        # Dry run: save the intended trade state for later review.
        self.memory.data["open_trade"] = {
            "symbol": symbol,
            "side": side,
            "score": float(signal["final_score"]),
            "raw_score": float(signal["raw_score"]),
            "lev": float(params["lev"]),
            "size_usdt": float(amount_usdt),
            "amount_base": float(amount_base),
            "entry_price": float(price),
            "sl_price": float(sl_price),
            "tp_price": float(tp_price),
            "entry_equity": float(self.memory.get_equity()),
            "regime": signal["regime"],
            "dominant_strategy": signal["dominant_strategy"],
            "hour": int(signal["hour"]),
            "opened_at": datetime.now().isoformat(),
            "mode": "DRY_RUN",
            "status": "open",
        }
        self.memory.save()
        return

    # Live execution path.
    self._attempt_set_leverage(symbol, params["lev"])
    order = self._place_live_entry(symbol, side, amount_base, sl_price, tp_price)
    if not order:
        send_telegram(f"⚠️ Entry failed for {symbol}")
        return

    self.memory.data["open_trade"] = {
        "symbol": symbol,
        "side": side,
        "score": float(signal["final_score"]),
        "raw_score": float(signal["raw_score"]),
        "lev": float(params["lev"]),
        "size_usdt": float(amount_usdt),
        "amount_base": float(amount_base),
        "entry_price": float(price),
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "entry_equity": float(self.memory.get_equity()),
        "regime": signal["regime"],
        "dominant_strategy": signal["dominant_strategy"],
        "hour": int(signal["hour"]),
        "opened_at": datetime.now().isoformat(),
        "entry_order_id": order.get("id"),
        "mode": "LIVE",
        "status": "open",
    }
    self.memory.data["last_trade_symbol"] = symbol
    self.memory.data["last_trade_side"] = side
    self.memory.save()

def run(self) -> None:
    self.memory.sync_equity_from_exchange(self.exchange)
    self._update_trade_day()

    if not self._survival_gate():
        return

    # First manage any open position from a prior run.
    self._manage_open_trade()
    if self.memory.data.get("open_trade"):
        # Only one open trade at a time.
        return

    try:
        symbols = get_top_symbols(self.exchange, MAX_SYMBOLS)
    except Exception as exc:
        send_telegram(f"⚠️ Failed to fetch symbols: {exc}")
        return

    if not symbols:
        send_telegram("No futures symbols found.")
        return

    best_signal = None
    best_df = None

    for sym in symbols:
        try:
            df = fetch_ohlcv(self.exchange, sym, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
            if df is None or len(df) < 50:
                continue

            signal = self._combine_signal(df, sym)
            if not signal:
                continue
            if signal["direction"] == "neutral":
                continue

            threshold = BASE_THRESHOLD + signal["bias"]
            if signal["hour"] in self.memory.data.get("toxic_hours", []):
                threshold += 0.08

            if signal["final_score"] < max(threshold, MIN_SIGNAL_SCORE):
                continue

            if best_signal is None or signal["final_score"] > best_signal["final_score"]:
                best_signal = signal
                best_df = df

        except Exception as exc:
            print(f"Error on {sym}: {exc}")

        time.sleep(0.25)

    if not best_signal or best_df is None:
        print("No high-confidence signal.")
        return

    self._enter_trade(best_signal, best_df)

======================================================

PAPER / MANUAL RESULT LOGGER

======================================================

def log_manual_trade_result( memory_file: str, symbol: str, side: str, score: float, lev: float, pnl_pct: float, strategy_used: str, regime: str, ): mem = Memory(memory_file) mem.record_trade( symbol=symbol, side=side, score=score, lev=lev, pnl_pct=pnl_pct, strategy_used=strategy_used, regime=regime, hour=datetime.now().hour, )

======================================================

ENTRY POINT

======================================================

if name == "main": ai = CozyHybridAI() ai.run()
