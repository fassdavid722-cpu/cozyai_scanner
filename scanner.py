#!/usr/bin/env python3
"""
CozyHybridAI — Production‑Ready Adaptive Trading Engine
- Persistent memory (cozy_memory.json stored in Google Drive)
- Ensemble of 5 strategies with adaptive weighting
- Self‑learning memory (toxic hours, symbol biases, strategy performance)
- Full risk management (daily loss, consecutive losses, timeout, leverage cap)
- Realistic dry‑run simulation (price‑based PnL, friction model)
- Live execution with attached SL/TP and emergency close
- Google Drive logging (raw_setups.csv, trade_log.csv, cozy_memory.json)
- Telegram alerts
- Fixed: minimum leverage 5x, minimum notional $5 for live trading
- NEW: Heartbeat, no‑setup explanations, startup/crash alerts, market pulse
"""

import os
import json
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from io import BytesIO

import ccxt
import numpy as np
import pandas as pd
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

print("DEBUG: Script started")
print(f"DEBUG: Python version: {sys.version}")

# ======================================================
# ENV / CONFIG (set these in GitHub Secrets or .env)
# ======================================================
BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_SECRET = os.getenv("BITGET_SECRET", "")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
USE_DEMO_TRADING = os.getenv("USE_DEMO_TRADING", "true").lower() == "true"

STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", "3.0"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "10.0"))
MIN_LEVERAGE = 5.0
MIN_NOTIONAL_EXCHANGE = 5.0
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.02"))
BOOTSTRAP_THRESHOLD = float(os.getenv("BOOTSTRAP_THRESHOLD", "10.0"))
BASE_THRESHOLD = float(os.getenv("BASE_THRESHOLD", "0.68"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "30"))
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "100"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "200"))
TAKE_PROFIT_R = float(os.getenv("TAKE_PROFIT_R", "2.0"))
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.65"))
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))

# Friction model
SPREAD_PCT = float(os.getenv("SPREAD_PCT", "0.0005"))
TAKER_FEE_PCT = float(os.getenv("TAKER_FEE_PCT", "0.0006"))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0003"))

# Volatility filter (skip if ATR/price < MIN_ATR_PCT)
MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0005"))

# Confidence decay for repeated symbols
CONFIDENCE_DECAY = float(os.getenv("CONFIDENCE_DECAY", "0.05"))

# Heartbeat and reporting (number of runs between messages)
HEARTBEAT_INTERVAL_RUNS = int(os.getenv("HEARTBEAT_INTERVAL_RUNS", "6"))   # ~1 hour if runs every 10 min
MARKET_PULSE_INTERVAL_RUNS = int(os.getenv("MARKET_PULSE_INTERVAL_RUNS", "12")) # ~2 hours

STRATEGY_LIST = [
    "volume_breakout",
    "orderbook_imbalance",
    "funding_arbitrage",
    "liquidity_sweep",
    "momentum",
]
DEFAULT_STRATEGY_WEIGHTS = {s: 1.0 / len(STRATEGY_LIST) for s in STRATEGY_LIST}
MEMORY_FILE = "cozy_memory.json"

# Google Drive settings
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1Ox77rDeIj7XEE_pyfE5TyKVtiYtbdcXe")
GDRIVE_CREDS_JSON = os.getenv("GDRIVE_CREDS", "")          # MUST be set in GitHub Secrets
LOG_TO_DRIVE = bool(GDRIVE_CREDS_JSON)

print(f"DEBUG: GDRIVE_CREDS present = {bool(GDRIVE_CREDS_JSON)}")
print(f"DEBUG: LOG_TO_DRIVE = {LOG_TO_DRIVE}")
print(f"DEBUG: DRIVE_FOLDER_ID = {DRIVE_FOLDER_ID}")

# ======================================================
# GOOGLE DRIVE HELPERS
# ======================================================
def get_drive_service():
    if not GDRIVE_CREDS_JSON:
        return None
    try:
        creds_dict = json.loads(GDRIVE_CREDS_JSON)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        print(f"Drive auth error: {e}")
        return None

def download_file(drive, filename, folder_id):
    if not drive:
        return None
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        res = drive.files().list(q=query, fields="files(id)").execute()
        files = res.get('files', [])
        if not files:
            return None
        file_id = files[0]['id']
        file_meta = drive.files().get(fileId=file_id, fields="mimeType").execute()
        mime_type = file_meta.get('mimeType')
        if mime_type == 'application/vnd.google-apps.spreadsheet':
            request = drive.files().export_media(fileId=file_id, mimeType='text/csv')
        else:
            request = drive.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh
    except Exception as e:
        print(f"Download error {filename}: {e}")
        return None

def upload_file(drive, filename, folder_id, content_bytes):
    if not drive:
        return
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        res = drive.files().list(q=query, fields="files(id)").execute()
        files = res.get('files', [])
        media = MediaIoBaseUpload(BytesIO(content_bytes), mimetype='application/json', resumable=True)
        if files:
            drive.files().update(fileId=files[0]['id'], media_body=media).execute()
        else:
            metadata = {'name': filename, 'parents': [folder_id]}
            drive.files().create(body=metadata, media_body=media).execute()
    except Exception as e:
        print(f"Upload error {filename}: {e}")

def append_to_csv(drive, filename, folder_id, new_rows_df):
    if not drive or new_rows_df.empty:
        return
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        res = drive.files().list(q=query, fields="files(id)").execute()
        files = res.get('files', [])
        if files:
            file_id = files[0]['id']
            request = drive.files().get_media(fileId=file_id)
            fh = BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            fh.seek(0)
            old_df = pd.read_csv(fh)
            df = pd.concat([old_df, new_rows_df], ignore_index=True)
        else:
            df = new_rows_df
        buf = BytesIO()
        df.to_csv(buf, index=False)
        upload_file(drive, filename, folder_id, buf.getvalue())
    except Exception as e:
        print(f"Append to Drive error: {e}")

# ======================================================
# HELPERS
# ======================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def safe_sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))

def send_telegram(message: str) -> None:
    print(message)
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": message,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
    except Exception as exc:
        print(f"Telegram error: {exc}")

def build_exchange() -> ccxt.bitget:
    exchange = ccxt.bitget(
        {
            "apiKey": BITGET_API_KEY,
            "secret": BITGET_SECRET,
            "password": BITGET_PASSWORD,
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
            "timeout": 20000,
        }
    )
    if USE_DEMO_TRADING:
        try:
            exchange.enable_demo_trading(True)
        except Exception:
            pass
    try:
        if DRY_RUN:
            exchange.set_sandbox_mode(True)
    except Exception:
        pass
    return exchange

def safe_call(func, retries: int = 3, delay: float = 1.5):
    last_exc = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            time.sleep(delay)
    if last_exc:
        raise last_exc
    return None

def apply_friction(price: float, is_entry: bool) -> float:
    friction = SPREAD_PCT + TAKER_FEE_PCT + SLIPPAGE_PCT
    if is_entry:
        return price * (1 + friction)
    else:
        return price * (1 - friction)

# ======================================================
# MEMORY (with Drive persistence)
# ======================================================
class Memory:
    def __init__(self, drive, folder_id):
        self.drive = drive
        self.folder_id = folder_id
        self.file = MEMORY_FILE
        self.data = self.load()
        self.normalize()

    def load(self) -> Dict:
        # Try to load from Drive first
        if self.drive:
            fh = download_file(self.drive, self.file, self.folder_id)
            if fh:
                try:
                    return json.load(fh)
                except Exception as e:
                    print(f"Error loading memory from Drive: {e}")
        # Fallback to local file or default
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
            "strategy_performance": {s: {"wins": 0, "losses": 0, "total_r": 0.0} for s in STRATEGY_LIST},
            "strategy_weights": DEFAULT_STRATEGY_WEIGHTS.copy(),
            "last_trade_symbol": None,
            "last_trade_side": None,
            "open_trade": None,
            "symbol_repeat": {},
            "heartbeat": {
                "last_run": None,
                "run_count": 0,
                "last_heartbeat_sent": None,
                "last_market_pulse_sent": None,
            },
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
        self.data.setdefault("strategy_performance", {s: {"wins": 0, "losses": 0, "total_r": 0.0} for s in STRATEGY_LIST})
        self.data.setdefault("strategy_weights", DEFAULT_STRATEGY_WEIGHTS.copy())
        self.data.setdefault("last_trade_symbol", None)
        self.data.setdefault("last_trade_side", None)
        self.data.setdefault("open_trade", None)
        self.data.setdefault("symbol_repeat", {})
        self.data.setdefault("heartbeat", {
            "last_run": None,
            "run_count": 0,
            "last_heartbeat_sent": None,
            "last_market_pulse_sent": None,
        })

    def save(self) -> None:
        # Save locally
        with open(self.file, "w") as f:
            json.dump(self.data, f, indent=4)
        # Upload to Drive
        if self.drive:
            buf = BytesIO()
            buf.write(json.dumps(self.data, indent=4).encode())
            buf.seek(0)
            upload_file(self.drive, self.file, self.folder_id, buf.getvalue())

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

    def record_daily_loss(self, loss_pct_account: float) -> None:
        self.data["daily_loss"] = float(self.data.get("daily_loss", 0.0)) + loss_pct_account
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

        self.data["symbol_repeat"][symbol] = self.data["symbol_repeat"].get(symbol, 0) + 1

        self.apply_learning()
        self.save()

    def apply_learning(self) -> None:
        history = self.data.get("performance_history", [])
        if len(history) < 10:
            return
        df = pd.DataFrame(history)
        hour_perf = df.groupby("hour")["pnl_pct"].mean()
        self.data["toxic_hours"] = [int(h) for h in hour_perf[hour_perf < -0.5].index.tolist()]
        sym_perf = df.groupby("symbol")["pnl_pct"].mean()
        for sym, avg in sym_perf.items():
            if avg < -1.0:
                self.data["learned_biases"][sym] = clamp(
                    self.data["learned_biases"].get(sym, 0.0) + 0.05,
                    0.0,
                    0.25,
                )
        for sym in list(self.data["learned_biases"].keys()):
            self.data["learned_biases"][sym] = max(0.0, self.data["learned_biases"][sym] - 0.02)
            if self.data["learned_biases"][sym] == 0:
                del self.data["learned_biases"][sym]
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

    # Heartbeat helper
    def update_heartbeat(self, run_count: int) -> None:
        self.data["heartbeat"]["last_run"] = datetime.utcnow().isoformat()
        self.data["heartbeat"]["run_count"] = run_count
        self.save()

    def get_last_run(self) -> Optional[str]:
        return self.data["heartbeat"].get("last_run")

# ======================================================
# MARKET DATA & STRATEGIES
# ======================================================
def fetch_ohlcv(exchange, symbol: str, timeframe: str = TIMEFRAME, limit: int = OHLCV_LIMIT) -> Optional[pd.DataFrame]:
    try:
        candles = safe_call(lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as exc:
        print(f"OHLCV error {symbol}: {exc}")
        return None

def fetch_order_book_imbalance(exchange, symbol: str) -> float:
    try:
        book = safe_call(lambda: exchange.fetch_order_book(symbol, limit=10))
        if not book:
            return 1.0
        bid_vol = sum(level[1] for level in (book.get("bids") or [])[:5])
        ask_vol = sum(level[1] for level in (book.get("asks") or [])[:5])
        if ask_vol <= 0:
            return 1.0
        return float(bid_vol / ask_vol)
    except Exception:
        return 1.0

def fetch_funding_rate(exchange, symbol: str) -> float:
    try:
        funding = safe_call(lambda: exchange.fetch_funding_rate(symbol))
        return float((funding or {}).get("fundingRate", 0.0))
    except Exception:
        return 0.0

def get_top_symbols(exchange, limit: int = MAX_SYMBOLS) -> List[str]:
    tickers = safe_call(lambda: exchange.fetch_tickers()) or {}
    df = pd.DataFrame.from_dict(tickers, orient="index")
    if df.empty or "symbol" not in df.columns:
        return []
    df = df[df["symbol"].fillna("").str.contains(":USDT")]
    if "quoteVolume" in df.columns:
        df = df.nlargest(limit, "quoteVolume")
    return df.index.tolist()[:limit]

def compute_volume_breakout(df: pd.DataFrame) -> Tuple[float, str]:
    if len(df) < 20:
        return 0.5, "neutral"
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

def compute_orderbook_imbalance(imbalance: float) -> Tuple[float, str]:
    if imbalance > 1.5:
        return min(0.9, (imbalance - 1.5) / 2.0 + 0.6), "long"
    if imbalance < 0.67:
        return min(0.9, (0.67 - imbalance) / 0.5 + 0.6), "short"
    return 0.5, "neutral"

def compute_funding_arbitrage(funding_rate: float) -> Tuple[float, str]:
    if funding_rate < -0.001:
        return 0.85, "long"
    if funding_rate > 0.001:
        return 0.85, "short"
    return 0.5, "neutral"

def compute_liquidity_sweep(df: pd.DataFrame) -> Tuple[float, str]:
    if len(df) < 30:
        return 0.5, "neutral"
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

def compute_momentum(df: pd.DataFrame) -> Tuple[float, str]:
    if len(df) < 26:
        return 0.5, "neutral"
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

def detect_regime(df: pd.DataFrame) -> str:
    if len(df) < 30:
        return "ranging"
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

def estimate_stop_distance_pct(df: pd.DataFrame) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    price = float(df["close"].iloc[-1])
    return max((atr * 1.2) / price, 0.006)

def get_atr_pct(df: pd.DataFrame) -> float:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()],
        axis=1
    ).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    price = float(df["close"].iloc[-1])
    return atr / price if price > 0 else 0

def compute_volume_spike(df: pd.DataFrame) -> float:
    """Volume spike ratio (last volume / average of last 20)."""
    if len(df) < 20:
        return 1.0
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    last_vol = df['volume'].iloc[-1]
    return last_vol / avg_vol if avg_vol > 0 else 1.0

# ======================================================
# SIZING (with minimum leverage and notional)
# ======================================================
def compute_position_size(equity: float, confidence: float, recent_trades: List[Dict]) -> float:
    confidence = clamp(confidence, 0.0, 1.0)
    risk_pct = RISK_PER_TRADE_PCT * confidence
    if len(recent_trades) >= 10:
        wins = [t["pnl_pct"] for t in recent_trades if t.get("pnl_pct", 0) > 0]
        losses = [abs(t["pnl_pct"]) for t in recent_trades if t.get("pnl_pct", 0) < 0]
        win_rate = len(wins) / max(1, len(recent_trades))
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 1.0
        if avg_win > 0 and avg_loss > 0:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly = clamp(kelly, 0.0, 0.10)
            risk_pct = min(risk_pct * (1.0 + kelly), 0.03)
    risk_amount = equity * risk_pct
    return max(MIN_NOTIONAL_EXCHANGE, float(risk_amount))

# ======================================================
# MAIN ENGINE (with heartbeat and reporting)
# ======================================================
class CozyHybridAI:
    def __init__(self):
        self.drive = get_drive_service() if LOG_TO_DRIVE else None
        self.memory = Memory(self.drive, DRIVE_FOLDER_ID)
        self.exchange = build_exchange()
        try:
            self.exchange.load_markets()
        except Exception:
            pass
        self.run_counter = 0
        self.rejection_reasons = []   # accumulate reasons for no-setup summary

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

    def _emergency_close(self, symbol: str, side: str, amount: float) -> None:
        try:
            close_side = "sell" if side == "long" else "buy"
            self.exchange.create_order(symbol, "market", close_side, amount, None)
            send_telegram(f"⚠️ Emergency closed {symbol} (timeout)")
        except Exception as e:
            send_telegram(f"❌ Emergency close failed {symbol}: {e}")

    def _log_closed_trade(self, trade: Dict, pnl_pct: float, account_loss_pct: float = None) -> None:
        if not self.drive:
            return
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": trade["symbol"],
            "side": trade["side"],
            "entry_price": trade.get("entry_price", 0.0),
            "sl_price": trade.get("sl_price", 0.0),
            "tp_price": trade.get("tp_price", 0.0),
            "pnl_pct": pnl_pct,
            "equity": self.memory.get_equity(),
            "strategy": trade.get("dominant_strategy", "ensemble"),
            "regime": trade.get("regime", "unknown"),
            "account_loss_pct": account_loss_pct if account_loss_pct is not None else 0.0,
        }])
        append_to_csv(self.drive, "trade_log.csv", DRIVE_FOLDER_ID, row)

    def _finalize_closed_trade(self, open_trade: Dict, pnl_pct: float, account_loss_pct: float = None) -> None:
        symbol = open_trade["symbol"]
        entry_equity = float(open_trade.get("entry_equity", self.memory.get_equity()))
        new_equity = entry_equity * (1 + pnl_pct / 100)
        self.memory.data["equity"] = new_equity

        self.memory.record_trade(
            symbol=symbol,
            side=open_trade.get("side", "long"),
            score=float(open_trade.get("score", 0.0)),
            lev=float(open_trade.get("lev", 1.0)),
            pnl_pct=pnl_pct,
            strategy_used=open_trade.get("dominant_strategy", "ensemble"),
            regime=open_trade.get("regime", "unknown"),
            hour=int(open_trade.get("hour", datetime.now().hour)),
        )

        if pnl_pct < 0:
            self.memory.record_loss(symbol)
            loss_pct_account = account_loss_pct if account_loss_pct is not None else (abs(pnl_pct) / 100)
            self.memory.record_daily_loss(loss_pct_account)
        else:
            self.memory.reset_consecutive_losses()

        self._log_closed_trade(open_trade, pnl_pct, account_loss_pct)

        self.memory.data["open_trade"] = None
        self.memory.data["last_trade_symbol"] = symbol
        self.memory.data["last_trade_side"] = open_trade.get("side")
        self.memory.save()

        send_telegram(
            f"📘 Trade closed\n"
            f"Symbol: {symbol}\n"
            f"PnL: {pnl_pct:.2f}%\n"
            f"Equity: ${new_equity:.2f}"
        )

    def _manage_open_trade(self) -> None:
        open_trade = self.memory.data.get("open_trade")
        if not open_trade:
            return

        if open_trade.get("mode") == "DRY_RUN":
            symbol = open_trade["symbol"]
            df = fetch_ohlcv(self.exchange, symbol, timeframe=TIMEFRAME, limit=5)
            if df is None or df.empty:
                return
            latest_price = float(df["close"].iloc[-1])
            side = open_trade["side"]
            entry_price = float(open_trade["entry_price"])
            sl_price = float(open_trade["sl_price"])
            tp_price = float(open_trade["tp_price"])
            lev = float(open_trade.get("lev", 1.0))

            if side == "long":
                raw_move = (latest_price - entry_price) / entry_price
                if latest_price <= sl_price:
                    raw_move = (sl_price - entry_price) / entry_price
                elif latest_price >= tp_price:
                    raw_move = (tp_price - entry_price) / entry_price
            else:
                raw_move = (entry_price - latest_price) / entry_price
                if latest_price >= sl_price:
                    raw_move = (entry_price - sl_price) / entry_price
                elif latest_price <= tp_price:
                    raw_move = (entry_price - tp_price) / entry_price

            pnl_pct = raw_move * lev * 100
            pnl_pct -= (SPREAD_PCT + TAKER_FEE_PCT + SLIPPAGE_PCT) * 100

            opened_at = datetime.fromisoformat(open_trade["opened_at"])
            age_minutes = (datetime.now() - opened_at).total_seconds() / 60
            if age_minutes > MAX_HOLD_MINUTES and not (latest_price <= sl_price or latest_price >= tp_price):
                pnl_pct = 0.0
                send_telegram(f"⏳ Dry‑run timeout exit for {symbol}")

            entry_equity = float(open_trade["entry_equity"])
            realized_dollar = entry_equity * (pnl_pct / 100)
            account_loss_pct = abs(realized_dollar) / entry_equity if realized_dollar < 0 else 0.0

            self._finalize_closed_trade(open_trade, pnl_pct, account_loss_pct)
            return

        # LIVE TRADE
        symbol = open_trade["symbol"]
        pos = self._position_from_exchange(symbol)
        if pos is not None:
            opened_at = datetime.fromisoformat(open_trade["opened_at"])
            age_minutes = (datetime.now() - opened_at).total_seconds() / 60
            if age_minutes > MAX_HOLD_MINUTES:
                amount = float(open_trade["amount_base"])
                side = open_trade["side"]
                self._emergency_close(symbol, side, amount)
                time.sleep(2)
                self._finalize_closed_trade(open_trade, 0.0, 0.0)
            return
        self._finalize_closed_trade(open_trade, 0.0, 0.0)

    def _combine_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Dict]:
        atr_pct = get_atr_pct(df)
        if atr_pct < MIN_ATR_PCT:
            self.rejection_reasons.append("ATR too low")
            return None

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
            self.rejection_reasons.append("No strategy weight")
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
        repeat_count = self.memory.data.get("symbol_repeat", {}).get(symbol, 0)
        penalty *= (1 - CONFIDENCE_DECAY * repeat_count)
        if hour in self.memory.data.get("toxic_hours", []):
            penalty -= 0.08

        final_score = clamp(combined_score * max(0.1, penalty), 0.0, 1.0)
        dominant_strategy = max(
            strategies,
            key=lambda x: x[0] * float(weights.get(x[2], 0.1)),
        )[2] if strategies else "ensemble"

        # Also check volume spike and sweep/FVG (these are already in strategies, but we add reasons)
        volume_spike = compute_volume_spike(df)
        if volume_spike < MIN_VOLUME_SPIKE:
            self.rejection_reasons.append("Volume spike too low")
            return None

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
        min_notional_exchange = MIN_NOTIONAL_EXCHANGE
        min_leverage = MIN_LEVERAGE

        if equity < BOOTSTRAP_THRESHOLD:
            mode = "BOOTSTRAP"
            size = max(MIN_NOTIONAL_EXCHANGE, min_notional_exchange)
            lev = max(min_leverage, size / max(equity, 0.01))
        else:
            mode = "PROFESSIONAL"
            ideal_size = (equity * RISK_PER_TRADE_PCT) / stop_dist_pct
            size = max(MIN_NOTIONAL_EXCHANGE, ideal_size, min_notional_exchange)
            lev = max(1.0, size / max(equity, 0.01))
            lev = max(lev, min_leverage)

        lev = clamp(lev, min_leverage, MAX_LEVERAGE)
        size = max(size, min_notional_exchange)

        return {"mode": mode, "size": round(size, 4), "lev": round(lev, 2)}

    def _place_live_entry(self, symbol: str, side: str, amount: float, sl_price: float, tp_price: float):
        order_side = "buy" if side == "long" else "sell"
        params = {
            "stopLossPrice": sl_price,
            "takeProfitPrice": tp_price,
        }
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

    def _log_raw_setup(self, signal: Dict, params: Dict, price: float, stop_dist_pct: float) -> None:
        if not self.drive:
            return
        row = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "price": price,
            "confidence": signal["final_score"],
            "raw_score": signal["raw_score"],
            "regime": signal["regime"],
            "dominant_strategy": signal["dominant_strategy"],
            "bias": signal["bias"],
            "mode": params["mode"],
            "size_usdt": params["size"],
            "leverage": params["lev"],
            "stop_distance_pct": stop_dist_pct,
            "hour": signal["hour"],
            "atr_pct": get_atr_pct(df) if 'df' in locals() else 0,
        }])
        append_to_csv(self.drive, "raw_setups.csv", DRIVE_FOLDER_ID, row)

    def _enter_trade(self, signal: Dict, df: pd.DataFrame) -> None:
        if LIVE_TRADING and DRY_RUN:
            send_telegram("❌ CONFIG ERROR: both LIVE_TRADING and DRY_RUN enabled")
            return

        symbol = signal["symbol"]
        side = signal["direction"]
        price = float(df["close"].iloc[-1])
        effective_entry = apply_friction(price, is_entry=True)
        stop_dist_pct = estimate_stop_distance_pct(df)
        params = self._calculate_trade_params(stop_dist_pct)
        amount_usdt = params["size"]
        amount_base = amount_usdt / effective_entry
        try:
            amount_base = float(self.exchange.amount_to_precision(symbol, amount_base))
        except Exception:
            amount_base = float(amount_base)
        if amount_base <= 0:
            send_telegram(f"⚠️ Amount precision invalid for {symbol}")
            return

        if side == "long":
            sl_price = effective_entry * (1.0 - stop_dist_pct)
            tp_price = effective_entry * (1.0 + stop_dist_pct * TAKE_PROFIT_R)
        else:
            sl_price = effective_entry * (1.0 + stop_dist_pct)
            tp_price = effective_entry * (1.0 - stop_dist_pct * TAKE_PROFIT_R)

        self._log_raw_setup(signal, params, effective_entry, stop_dist_pct)

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
            f"Entry (friction adj): {effective_entry:.6f}\n"
            f"SL: {sl_price:.6f}\n"
            f"TP: {tp_price:.6f}"
        )
        send_telegram(msg)

        if DRY_RUN or not LIVE_TRADING:
            self.memory.data["open_trade"] = {
                "symbol": symbol,
                "side": side,
                "score": float(signal["final_score"]),
                "raw_score": float(signal["raw_score"]),
                "lev": float(params["lev"]),
                "size_usdt": float(amount_usdt),
                "amount_base": float(amount_base),
                "entry_price": effective_entry,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "entry_equity": float(self.memory.get_equity()),
                "regime": signal["regime"],
                "dominant_strategy": signal["dominant_strategy"],
                "hour": int(signal["hour"]),
                "opened_at": datetime.now().isoformat(),
                "mode": "DRY_RUN",
                "status": "open",
                "max_hold_minutes": MAX_HOLD_MINUTES,
            }
            self.memory.save()
            return

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
            "entry_price": effective_entry,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "entry_equity": float(self.memory.get_equity()),
            "regime": signal["regime"],
            "dominant_strategy": signal["dominant_strategy"],
            "hour": int(signal["hour"]),
            "opened_at": datetime.now().isoformat(),
            "entry_order_id": order.get("id"),
            "mode": "LIVE",
            "status": "open",
            "max_hold_minutes": MAX_HOLD_MINUTES,
        }
        self.memory.data["last_trade_symbol"] = symbol
        self.memory.data["last_trade_side"] = side
        self.memory.save()

    def _send_heartbeat(self):
        last_run = self.memory.get_last_run()
        last_run_str = last_run if last_run else "never"
        msg = (
            f"🟢 CozyHybridAI Heartbeat\n"
            f"Time: {datetime.utcnow().isoformat()}\n"
            f"Run count: {self.run_counter}\n"
            f"Last run: {last_run_str}\n"
            f"Symbols scanned: {MAX_SYMBOLS}\n"
            f"Equity: ${self.memory.get_equity():.2f}\n"
            f"Open trade: {self.memory.data.get('open_trade') is not None}\n"
            f"Regime: {self.last_regime if hasattr(self, 'last_regime') else 'unknown'}"
        )
        send_telegram(msg)
        self.memory.update_heartbeat(self.run_counter)

    def _send_no_setup_summary(self):
        if not self.rejection_reasons:
            return
        # Count reasons
        from collections import Counter
        reason_counts = Counter(self.rejection_reasons)
        msg = f"⚠️ No setup found after scanning {MAX_SYMBOLS} symbols.\nTop reasons:\n"
        for reason, count in reason_counts.most_common(5):
            msg += f"  - {reason}: {count}\n"
        send_telegram(msg)

    def _send_market_pulse(self):
        msg = (
            f"📊 Market Pulse\n"
            f"Scans since last pulse: {HEARTBEAT_INTERVAL_RUNS}\n"
            f"Equity: ${self.memory.get_equity():.2f}\n"
            f"Open trade: {self.memory.data.get('open_trade') is not None}\n"
            f"Last setup found: {self.memory.data.get('last_trade_symbol', 'none')}"
        )
        send_telegram(msg)

    def run(self) -> None:
        print("DEBUG: CozyHybridAI.run() started")
        # Send startup alert
        send_telegram("🟢 Trading AI started successfully")
        try:
            self.memory.sync_equity_from_exchange(self.exchange)
            self._update_trade_day()
            if not self._survival_gate():
                return
            self._manage_open_trade()
            if self.memory.data.get("open_trade"):
                print("DEBUG: Open trade exists, skipping new signal")
                return

            # Increment run counter and update heartbeat
            self.run_counter = self.memory.data["heartbeat"]["run_count"] + 1
            self.memory.update_heartbeat(self.run_counter)

            # Fetch symbols and scan
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
            self.rejection_reasons = []

            for sym in symbols:
                try:
                    df = fetch_ohlcv(self.exchange, sym, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
                    if df is None or len(df) < 50:
                        self.rejection_reasons.append("OHLCV data insufficient")
                        continue

                    signal = self._combine_signal(df, sym)
                    if not signal:
                        continue
                    if signal["direction"] == "neutral":
                        self.rejection_reasons.append("Direction neutral")
                        continue

                    threshold = BASE_THRESHOLD + signal["bias"]
                    if signal["hour"] in self.memory.data.get("toxic_hours", []):
                        threshold += 0.08
                    if signal["final_score"] < max(threshold, MIN_SIGNAL_SCORE):
                        self.rejection_reasons.append(f"Score below threshold ({signal['final_score']:.2f} < {threshold:.2f})")
                        continue

                    if best_signal is None or signal["final_score"] > best_signal["final_score"]:
                        best_signal = signal
                        best_df = df
                        self.last_regime = signal["regime"]
                except Exception as exc:
                    print(f"Error on {sym}: {exc}")
                    self.rejection_reasons.append(f"Exception: {str(exc)[:50]}")
                time.sleep(0.25)

            if not best_signal or best_df is None:
                print("DEBUG: No high-confidence signal.")
                self._send_no_setup_summary()
                # Send heartbeat if interval reached
                if self.run_counter % HEARTBEAT_INTERVAL_RUNS == 0:
                    self._send_heartbeat()
                if self.run_counter % MARKET_PULSE_INTERVAL_RUNS == 0:
                    self._send_market_pulse()
                return

            print("DEBUG: Entering trade with best signal")
            self._enter_trade(best_signal, best_df)

        except Exception as e:
            send_telegram(f"🔴 Scanner crashed: {e}")
            raise

# ======================================================
# ENTRY POINT
# ======================================================
if __name__ == "__main__":
    print("DEBUG: Before creating AI instance")
    ai = CozyHybridAI()
    print("DEBUG: AI instance created, running")
    ai.run()
    print("DEBUG: Run finished")
