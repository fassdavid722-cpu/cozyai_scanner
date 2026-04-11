#!/usr/bin/env python3
"""
CozyHybridAI v3 — Institutional Scanner

Features:
- Bitget futures market scanner
- 5-strategy ensemble with adaptive weighting
- Liquidity / FVG / momentum / volume / orderbook / funding signals
- Trade lifecycle manager (open -> manage -> close)
- Logging (local + optional Google Drive sync)
- Dry-run + live trading support
- Heartbeat + stall watchdog
- Self-learning memory system

IMPORTANT:
- Start with DRY_RUN=true
- LIVE_TRADING=false initially
- Set all secrets via environment variables
"""

from __future__ import annotations

import os
import json
import time
import traceback
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import ccxt
import numpy as np
import pandas as pd

# Optional Google Drive imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
except Exception:
    service_account = None
    build = None
    MediaIoBaseUpload = None
    MediaIoBaseDownload = None

# ======================================================
# CONFIGURATION (environment variables)
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
MIN_LEVERAGE = float(os.getenv("MIN_LEVERAGE", "5.0"))
DEFAULT_LEVERAGE = float(os.getenv("DEFAULT_LEVERAGE", "5.0"))
MIN_NOTIONAL_EXCHANGE = float(os.getenv("MIN_NOTIONAL_EXCHANGE", "5.0"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.02"))
BOOTSTRAP_THRESHOLD = float(os.getenv("BOOTSTRAP_THRESHOLD", "10.0"))
BASE_THRESHOLD = float(os.getenv("BASE_THRESHOLD", "0.68"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "20"))
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "60"))
FAST_ENTRY_SCORE = float(os.getenv("FAST_ENTRY_SCORE", "0.78"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "200"))
TAKE_PROFIT_R = float(os.getenv("TAKE_PROFIT_R", "2.0"))
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.65"))
MAX_HOLD_MINUTES = int(os.getenv("MAX_HOLD_MINUTES", "30"))

SPREAD_PCT = float(os.getenv("SPREAD_PCT", "0.0005"))
TAKER_FEE_PCT = float(os.getenv("TAKER_FEE_PCT", "0.0006"))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", "0.0003"))

MIN_ATR_PCT = float(os.getenv("MIN_ATR_PCT", "0.0005"))
MIN_VOLUME_SPIKE = float(os.getenv("MIN_VOLUME_SPIKE", "0.0"))
CONFIDENCE_DECAY = float(os.getenv("CONFIDENCE_DECAY", "0.05"))

HEARTBEAT_INTERVAL_RUNS = int(os.getenv("HEARTBEAT_INTERVAL_RUNS", "6"))
MARKET_PULSE_INTERVAL_RUNS = int(os.getenv("MARKET_PULSE_INTERVAL_RUNS", "12"))
STALL_ALERT_AFTER_MINUTES = int(os.getenv("STALL_ALERT_AFTER_MINUTES", "15"))

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1Ox77rDeIj7XEE_pyfE5TyKVtiYtbdcXe")
GDRIVE_CREDS_JSON = os.getenv("GDRIVE_CREDS", "")
LOG_TO_DRIVE = bool(GDRIVE_CREDS_JSON)

MEMORY_FILE = os.getenv("COZY_MEMORY_FILE", "cozy_memory.json")
LOCAL_TRADE_LOG = os.getenv("LOCAL_TRADE_LOG", "trade_log_local.csv")
RAW_SETUPS_LOG = os.getenv("RAW_SETUPS_LOG", "raw_setups.csv")
LOG_FILE = os.getenv("ALPHA_LOG_FILE", "cozy_scanner.log")

STRATEGY_LIST = [
    "volume_breakout",
    "orderbook_imbalance",
    "funding_arbitrage",
    "liquidity_sweep",
    "momentum",
]
DEFAULT_STRATEGY_WEIGHTS = {s: 1.0 / len(STRATEGY_LIST) for s in STRATEGY_LIST}

Path(".").mkdir(parents=True, exist_ok=True)

# ======================================================
# LOGGING
# ======================================================
def log(msg: str) -> None:
    line = f"{datetime.utcnow().isoformat()} | {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ======================================================
# TELEGRAM
# ======================================================
def send_telegram(message: str) -> None:
    log(message)
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": message}, timeout=10)
    except Exception as exc:
        log(f"Telegram error: {exc}")

# ======================================================
# GOOGLE DRIVE HELPERS
# ======================================================
def get_drive_service():
    if not LOG_TO_DRIVE or service_account is None or build is None:
        return None
    try:
        creds_dict = json.loads(GDRIVE_CREDS_JSON)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict, scopes=["https://www.googleapis.com/auth/drive"]
        )
        return build("drive", "v3", credentials=creds)
    except Exception as exc:
        log(f"Drive auth error: {exc}")
        return None

def download_file(drive, filename: str, folder_id: str):
    if not drive or MediaIoBaseDownload is None:
        return None
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        res = drive.files().list(q=query, fields="files(id)").execute()
        files = res.get("files", [])
        if not files:
            return None
        file_id = files[0]["id"]
        request = drive.files().get_media(fileId=file_id)
        fh = BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return fh
    except Exception as exc:
        log(f"Download error {filename}: {exc}")
        return None

def upload_file(drive, filename: str, folder_id: str, content_bytes: bytes):
    if not drive or MediaIoBaseUpload is None:
        return
    try:
        query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
        res = drive.files().list(q=query, fields="files(id)").execute()
        files = res.get("files", [])
        media = MediaIoBaseUpload(BytesIO(content_bytes), mimetype="text/csv", resumable=True)
        if files:
            drive.files().update(fileId=files[0]["id"], media_body=media).execute()
        else:
            metadata = {"name": filename, "parents": [folder_id]}
            drive.files().create(body=metadata, media_body=media).execute()
    except Exception as exc:
        log(f"Upload error {filename}: {exc}")

def append_to_csv(drive, filename: str, folder_id: str, new_rows_df: pd.DataFrame):
    if new_rows_df is None or new_rows_df.empty:
        return
    # Local fallback
    try:
        header = not os.path.exists(filename)
        new_rows_df.to_csv(filename, mode="a", header=header, index=False)
    except Exception as exc:
        log(f"Local CSV append error for {filename}: {exc}")
    # Drive upload
    try:
        if drive and MediaIoBaseDownload is not None:
            query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
            res = drive.files().list(q=query, fields="files(id)").execute()
            files = res.get("files", [])
            if files:
                file_id = files[0]["id"]
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
    except Exception as exc:
        log(f"Append to Drive error for {filename}: {exc}")

# ======================================================
# HELPERS
# ======================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))

def safe_sigmoid(x: float) -> float:
    x = float(np.clip(x, -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(-x)))

def safe_call(func, retries: int = 3, delay: float = 1.5):
    last_exc = None
    for _ in range(retries):
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
    return price * (1 + friction) if is_entry else price * (1 - friction)

def make_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time())}"

# ======================================================
# MEMORY (persistent state)
# ======================================================
class Memory:
    def __init__(self, drive, folder_id: str):
        self.drive = drive
        self.folder_id = folder_id
        self.file = MEMORY_FILE
        self.data = self.load()
        self.normalize()

    def default_state(self) -> Dict[str, Any]:
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
            "best_candidate": None,
            "last_rejection": None,
            "system_summary": "",
            "recent_no_setup_reasons": [],
        }

    def load(self) -> Dict[str, Any]:
        if self.drive:
            fh = download_file(self.drive, self.file, self.folder_id)
            if fh:
                try:
                    return json.load(fh)
                except Exception as exc:
                    log(f"Error loading memory from Drive: {exc}")
        if os.path.exists(self.file):
            try:
                with open(self.file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self.default_state()

    def normalize(self) -> None:
        default = self.default_state()
        for k, v in default.items():
            self.data.setdefault(k, v)
        self.data.setdefault("heartbeat", default["heartbeat"])

    def save(self) -> None:
        try:
            with open(self.file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as exc:
            log(f"Local memory save failed: {exc}")
        if self.drive:
            try:
                buf = BytesIO()
                buf.write(json.dumps(self.data, indent=2).encode())
                buf.seek(0)
                upload_file(self.drive, self.file, self.folder_id, buf.getvalue())
            except Exception as exc:
                log(f"Drive memory save failed: {exc}")

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
        self.data["daily_loss"] = float(self.data.get("daily_loss", 0.0)) + float(loss_pct_account)
        self.save()

    def reset_consecutive_losses(self) -> None:
        self.data["consecutive_losses"] = 0
        self.save()

    def record_loss(self, symbol: str) -> None:
        self.data["sl_count"] = int(self.data.get("sl_count", 0)) + 1
        self.data["last_sl_time"] = datetime.now().isoformat()
        self.data["learned_biases"][symbol] = clamp(
            self.data["learned_biases"].get(symbol, 0.0) + 0.05, 0.0, 0.25
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
                    self.data["learned_biases"].get(sym, 0.0) + 0.05, 0.0, 0.25
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

    def update_heartbeat(self, run_count: int) -> None:
        self.data["heartbeat"]["last_run"] = datetime.utcnow().isoformat()
        self.data["heartbeat"]["run_count"] = int(run_count)
        self.save()

    def get_last_run(self) -> Optional[str]:
        return self.data.get("heartbeat", {}).get("last_run")

    def get_dynamic_threshold(self) -> float:
        history = self.data.get("performance_history", [])
        if len(history) < 20:
            return BASE_THRESHOLD
        df = pd.DataFrame(history)
        win_rate = (df["pnl_pct"] > 0).mean()
        if win_rate > 0.55:
            return BASE_THRESHOLD - 0.03
        if win_rate < 0.45:
            return BASE_THRESHOLD + 0.05
        return BASE_THRESHOLD

    def adjust_filters_based_on_performance(self) -> None:
        history = self.data.get("performance_history", [])
        if len(history) < 30:
            return
        df = pd.DataFrame(history)
        recent = df.tail(30)
        win_rate = (recent["pnl_pct"] > 0).mean()

        global MIN_SIGNAL_SCORE
        if win_rate < 0.4:
            MIN_SIGNAL_SCORE = min(0.75, MIN_SIGNAL_SCORE + 0.02)
        elif win_rate > 0.6:
            MIN_SIGNAL_SCORE = max(0.60, MIN_SIGNAL_SCORE - 0.02)

# ======================================================
# MARKET DATA
# ======================================================
class MarketData:
    def __init__(self, exchange):
        self.exchange = exchange

    def fetch_ohlcv(self, symbol: str, timeframe: str = TIMEFRAME, limit: int = OHLCV_LIMIT) -> Optional[pd.DataFrame]:
        try:
            candles = safe_call(lambda: self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit))
            if not candles:
                return None
            df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as exc:
            log(f"OHLCV error {symbol}: {exc}")
            return None

    def fetch_order_book_imbalance(self, symbol: str) -> float:
        try:
            book = safe_call(lambda: self.exchange.fetch_order_book(symbol, limit=10))
            if not book:
                return 1.0
            bid_vol = sum(level[1] for level in (book.get("bids") or [])[:5])
            ask_vol = sum(level[1] for level in (book.get("asks") or [])[:5])
            if ask_vol <= 0:
                return 1.0
            return float(bid_vol / ask_vol)
        except Exception:
            return 1.0

    def fetch_funding_rate(self, symbol: str) -> float:
        try:
            funding = safe_call(lambda: self.exchange.fetch_funding_rate(symbol))
            return float((funding or {}).get("fundingRate", 0.0))
        except Exception:
            return 0.0

    def get_top_symbols(self, limit: int = MAX_SYMBOLS) -> List[str]:
        tickers = safe_call(lambda: self.exchange.fetch_tickers()) or {}
        df = pd.DataFrame.from_dict(tickers, orient="index")
        if df.empty or "symbol" not in df.columns:
            return []
        df = df[df["symbol"].fillna("").str.contains(":USDT")]
        if "quoteVolume" in df.columns:
            df = df.nlargest(limit, "quoteVolume")
        return df.index.tolist()[:limit]

# ======================================================
# STRATEGY FUNCTIONS
# ======================================================
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
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
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
    return max((atr * 1.2) / price, MIN_ATR_PCT)

def get_atr_pct(df: pd.DataFrame) -> float:
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
    return float(atr / price) if price > 0 else 0.0

def compute_volume_spike(df: pd.DataFrame) -> float:
    if len(df) < 20:
        return 1.0
    avg_vol = df["volume"].rolling(20).mean().iloc[-1]
    last_vol = df["volume"].iloc[-1]
    return float(last_vol / avg_vol) if avg_vol > 0 else 1.0

def compute_liquidity_quality(df: pd.DataFrame, imbalance: float, volume_spike: float) -> float:
    score = 0.5
    atr_pct = get_atr_pct(df)
    if volume_spike > 1.5:
        score += 0.2
    elif volume_spike < 1.0:
        score -= 0.2
    if imbalance > 1.3 or imbalance < 0.7:
        score += 0.2
    else:
        score -= 0.1
    if 0.001 < atr_pct < 0.01:
        score += 0.2
    else:
        score -= 0.1
    return clamp(score, 0.0, 1.0)

# ======================================================
# DIAGNOSTICS
# ======================================================
class MarketNeedsAnalyzer:
    def analyze(self, df: pd.DataFrame, symbol: str, imbalance: float, atr_pct: float, volume_spike: float) -> List[str]:
        needs = []
        if atr_pct < 0.001:
            needs.append("Low volatility → waiting for liquidity expansion")
        if volume_spike < 1.2:
            needs.append("Weak volume participation → no institutional interest")
        body = abs(df["close"].iloc[-1] - df["open"].iloc[-1])
        candle_range = (df["high"].iloc[-1] - df["low"].iloc[-1]) + 1e-9
        if body / candle_range < 0.3:
            needs.append("Indecision candle → no directional displacement")
        if 0.8 < imbalance < 1.2:
            needs.append("Neutral orderbook → no directional pressure")
        return needs

# ======================================================
# INTENT MODEL (for future use)
# ======================================================
@dataclass
class TradeIntent:
    id: str
    symbol: str
    mode: str  # immediate | watch
    direction: str  # long | short
    trigger: str  # breakout | dump | momentum | sweep | breakdown | manual
    take_profit_pct: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    leverage: float = DEFAULT_LEVERAGE
    risk_level: str = "medium"
    note: str = ""
    created_at: str = ""
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ======================================================
# TRADE LIFECYCLE
# ======================================================
class TradeLifecycle:
    def __init__(self, memory: Memory, market: MarketData, exchange, drive):
        self.memory = memory
        self.market = market
        self.exchange = exchange
        self.drive = drive
        self._finalized_ids = set()

    def is_finalized(self, trade_id: str) -> bool:
        return trade_id in self._finalized_ids

    def mark_finalized(self, trade_id: str) -> None:
        self._finalized_ids.add(trade_id)

    def _save_closed_trade(self, trade: Dict[str, Any], pnl_pct: float, account_loss_pct: float = 0.0):
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_id": trade.get("id"),
            "symbol": trade["symbol"],
            "side": trade["side"],
            "entry_price": trade.get("entry_price", 0.0),
            "sl_price": trade.get("sl_price", 0.0),
            "tp_price": trade.get("tp_price", 0.0),
            "pnl_pct": pnl_pct,
            "equity": self.memory.get_equity(),
            "strategy": trade.get("dominant_strategy", "ensemble"),
            "regime": trade.get("regime", "unknown"),
            "account_loss_pct": account_loss_pct,
            "mode": trade.get("mode", "DRY_RUN"),
        }
        df = pd.DataFrame([row])
        # Local
        header = not os.path.exists(LOCAL_TRADE_LOG)
        df.to_csv(LOCAL_TRADE_LOG, mode="a", header=header, index=False)
        # Drive
        if self.drive:
            append_to_csv(self.drive, "trade_log.csv", DRIVE_FOLDER_ID, df)

    def finalize_closed_trade(self, trade: Dict[str, Any], pnl_pct: float, account_loss_pct: float = 0.0, reason: str = "close"):
        trade_id = trade.get("id")
        if trade_id and self.is_finalized(trade_id):
            return
        symbol = trade["symbol"]
        entry_equity = float(trade.get("entry_equity", self.memory.get_equity()))
        new_equity = entry_equity * (1 + pnl_pct / 100)
        self.memory.data["equity"] = new_equity

        self.memory.record_trade(
            symbol=symbol,
            side=trade.get("side", "long"),
            score=float(trade.get("score", 0.0)),
            lev=float(trade.get("lev", 1.0)),
            pnl_pct=float(pnl_pct),
            strategy_used=trade.get("dominant_strategy", "ensemble"),
            regime=trade.get("regime", "unknown"),
            hour=int(trade.get("hour", datetime.now().hour)),
        )

        if pnl_pct < 0:
            self.memory.record_loss(symbol)
            self.memory.record_daily_loss(account_loss_pct if account_loss_pct > 0 else abs(pnl_pct) / 100)
        else:
            self.memory.reset_consecutive_losses()

        self._save_closed_trade(trade, pnl_pct, account_loss_pct)
        self.memory.data["open_trade"] = None
        self.memory.data["last_trade_symbol"] = symbol
        self.memory.data["last_trade_side"] = trade.get("side")
        self.memory.save()
        if trade_id:
            self.mark_finalized(trade_id)

        send_telegram(
            f"📘 Trade closed ({reason})\n"
            f"Symbol: {symbol}\n"
            f"PnL: {pnl_pct:.2f}%\n"
            f"Equity: ${new_equity:.2f}"
        )

    def manage_open_trade(self):
        trade = self.memory.data.get("open_trade")
        if not trade:
            return

        trade_id = trade.get("id")
        if trade_id and self.is_finalized(trade_id):
            self.memory.data["open_trade"] = None
            self.memory.save()
            return

        symbol = trade["symbol"]
        df = self.market.fetch_ohlcv(symbol, limit=10)
        if df is None or df.empty:
            return

        price = float(df["close"].iloc[-1])
        side = trade["side"]
        entry = float(trade["entry_price"])
        sl = float(trade["sl_price"])
        tp = float(trade["tp_price"])

        candle_high = float(df["high"].max())
        candle_low = float(df["low"].min())
        hit_sl = candle_low <= sl if side == "long" else candle_high >= sl
        hit_tp = candle_high >= tp if side == "long" else candle_low <= tp

        if hit_sl:
            pnl_pct = -abs((entry - sl) / entry) * 100
            self.finalize_closed_trade(trade, pnl_pct, abs(pnl_pct) / 100, reason="SL")
            return
        if hit_tp:
            pnl_pct = abs((tp - entry) / entry) * 100
            self.finalize_closed_trade(trade, pnl_pct, 0.0, reason="TP")
            return

        opened_at = datetime.fromisoformat(trade["opened_at"])
        age_minutes = (datetime.now() - opened_at).total_seconds() / 60
        if age_minutes > MAX_HOLD_MINUTES:
            pnl_pct = ((price - entry) / entry) * 100 if side == "long" else ((entry - price) / entry) * 100
            account_loss_pct = abs(pnl_pct) / 100 if pnl_pct < 0 else 0.0
            send_telegram(f"⏳ Forced exit {symbol} (timeout)")
            self.finalize_closed_trade(trade, pnl_pct, account_loss_pct, reason="TIMEOUT")

    def validate_live_exit(self, symbol: str, side: str, amount: float):
        try:
            close_side = "sell" if side == "long" else "buy"
            self.exchange.create_order(symbol, "market", close_side, amount, None)
            send_telegram(f"⚠️ Emergency closed {symbol}")
        except Exception as exc:
            send_telegram(f"❌ Emergency close failed {symbol}: {exc}")

# ======================================================
# SIGNAL ENGINE
# ======================================================
class SignalEngine:
    def __init__(self, memory: Memory):
        self.memory = memory
        self.needs_analyzer = MarketNeedsAnalyzer()
        self.rejection_reasons: List[str] = []
        self.last_regime = "unknown"

    def _score_intel(self, intel: Dict[str, Any], direction: str) -> float:
        if not intel.get("valid"):
            return 0.0
        score = 0.0
        atr_pct = float(intel.get("atr_pct", 0.0))
        rsi = float(intel.get("rsi", 50.0))
        vol = float(intel.get("volume_spike", 1.0))
        trend = float(intel.get("trend_slope", 0.0))
        sweep = intel.get("sweep", {}) or {}
        breakout = intel.get("breakout", {}) or {}
        fvg = intel.get("fvg", {}) or {}

        if atr_pct >= MIN_ATR_PCT:
            score += 0.15
        if vol >= max(1.0, MIN_VOLUME_SPIKE):
            score += 0.15
        if direction == "long":
            if trend > 0:
                score += 0.20
            if rsi > 52:
                score += 0.10
            if sweep.get("long"):
                score += 0.20
            if breakout.get("breakout_long"):
                score += 0.15
            if fvg.get("type") == "bullish":
                score += 0.10
        else:
            if trend < 0:
                score += 0.20
            if rsi < 48:
                score += 0.10
            if sweep.get("short"):
                score += 0.20
            if breakout.get("breakout_short"):
                score += 0.15
            if fvg.get("type") == "bearish":
                score += 0.10
        return clamp(score, 0.0, 0.95)

    def _safe_df(self, df: Any) -> bool:
        return pd is not None and isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in ["open", "high", "low", "close", "volume"])

    def _calc_fvg(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 3:
            return {"type": None, "width": 0.0}
        c1 = df.iloc[-3]
        c3 = df.iloc[-1]
        if c1["high"] < c3["low"]:
            return {"type": "bullish", "width": float(c3["low"] - c1["high"]), "low": float(c1["high"]), "high": float(c3["low"])}
        if c1["low"] > c3["high"]:
            return {"type": "bearish", "width": float(c1["low"] - c3["high"]), "low": float(c3["high"]), "high": float(c1["low"])}
        return {"type": None, "width": 0.0}

    def _detect_trigger(self, df: pd.DataFrame, intent: TradeIntent) -> Tuple[bool, str, float, Dict[str, Any]]:
        if not self._safe_df(df):
            return False, "invalid dataframe", 0.0, {}

        intel = {
            "valid": True,
            "atr_pct": get_atr_pct(df),
            "rsi": self._rsi(df),
            "volume_spike": compute_volume_spike(df),
            "sweep": self._liquidity_sweep(df),
            "breakout": self._breakout_state(df),
            "fvg": self._calc_fvg(df),
            "trend_slope": self._trend_slope(df),
            "last_close": float(df["close"].iloc[-1]),
        }

        rsi = float(intel["rsi"])
        vol_spike = float(intel["volume_spike"])
        atr_pct = float(intel["atr_pct"])
        trend = float(intel["trend_slope"])
        sweep = intel["sweep"]
        breakout = intel["breakout"]
        fvg = intel["fvg"]

        reasons: List[str] = []

        if atr_pct < MIN_ATR_PCT:
            reasons.append(f"ATR {atr_pct:.4f} below minimum {MIN_ATR_PCT:.4f}")
        if vol_spike < max(1.0, MIN_VOLUME_SPIKE):
            reasons.append(f"Volume spike {vol_spike:.2f} below minimum")

        # trigger logic
        last = float(df["close"].iloc[-1])
        prev = float(df["close"].iloc[-2]) if len(df) > 1 else last

        if intent.trigger == "breakout":
            if intent.direction == "long" and breakout.get("breakout_long"):
                reasons.append("breakout_long")
                score = self._score_intel(intel, "long")
                return True, "breakout long confirmed", score, {"intel": intel, "reasons": reasons, "entry": last}
            if intent.direction == "short" and breakout.get("breakout_short"):
                reasons.append("breakout_short")
                score = self._score_intel(intel, "short")
                return True, "breakout short confirmed", score, {"intel": intel, "reasons": reasons, "entry": last}
            return False, "; ".join(reasons) or "waiting for breakout", self._score_intel(intel, intent.direction), {"intel": intel, "reasons": reasons}

        if intent.trigger == "dump":
            if intent.direction == "short" and last < prev and vol_spike >= max(1.0, MIN_VOLUME_SPIKE):
                reasons.append("dump_momentum")
                score = self._score_intel(intel, "short")
                return True, "dump short confirmed", score, {"intel": intel, "reasons": reasons, "entry": last}
            return False, "; ".join(reasons) or "waiting for dump confirmation", self._score_intel(intel, intent.direction), {"intel": intel, "reasons": reasons}

        if intent.trigger == "sweep":
            if intent.direction == "long" and sweep.get("long"):
                reasons.append("liquidity_sweep_long")
                score = self._score_intel(intel, "long")
                return True, "liquidity sweep long confirmed", score, {"intel": intel, "reasons": reasons, "entry": last}
            if intent.direction == "short" and sweep.get("short"):
                reasons.append("liquidity_sweep_short")
                score = self._score_intel(intel, "short")
                return True, "liquidity sweep short confirmed", score, {"intel": intel, "reasons": reasons, "entry": last}
            return False, "; ".join(reasons) or "waiting for sweep confirmation", self._score_intel(intel, intent.direction), {"intel": intel, "reasons": reasons}

        if intent.trigger == "momentum":
            if intent.direction == "long" and rsi > 52 and last > prev and trend >= 0:
                reasons.append("momentum_long")
                score = self._score_intel(intel, "long")
                return True, f"momentum long confirmed; RSI={rsi:.1f}", score, {"intel": intel, "reasons": reasons, "entry": last}
            if intent.direction == "short" and rsi < 48 and last < prev and trend <= 0:
                reasons.append("momentum_short")
                score = self._score_intel(intel, "short")
                return True, f"momentum short confirmed; RSI={rsi:.1f}", score, {"intel": intel, "reasons": reasons, "entry": last}
            return False, "; ".join(reasons) or f"waiting for momentum confirmation; RSI={rsi:.1f}", self._score_intel(intel, intent.direction), {"intel": intel, "reasons": reasons}

        # default/manual behavior
        if intent.direction == "long" and rsi > 52:
            reasons.append("default_long_rsi")
            score = self._score_intel(intel, "long")
            return True, "default long confirmed", score, {"intel": intel, "reasons": reasons, "entry": last}
        if intent.direction == "short" and rsi < 48:
            reasons.append("default_short_rsi")
            score = self._score_intel(intel, "short")
            return True, "default short confirmed", score, {"intel": intel, "reasons": reasons, "entry": last}

        return False, "; ".join(reasons) or "no trigger confluence", self._score_intel(intel, intent.direction), {"intel": intel, "reasons": reasons, "entry": last}

    def _rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        close = df["close"]
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        avg_up = up.rolling(period).mean().iloc[-1]
        avg_down = down.rolling(period).mean().iloc[-1]
        if avg_down == 0:
            return 100.0
        return float(100 - (100 / (1 + (avg_up / avg_down))))

    def _trend_slope(self, df: pd.DataFrame, length: int = 20) -> float:
        if len(df) < length:
            return 0.0
        x = np.arange(length)
        y = df["close"].tail(length).values
        try:
            return float(np.polyfit(x, y, 1)[0])
        except Exception:
            return 0.0

    def _liquidity_sweep(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 25:
            return {"long": False, "short": False, "reason": "insufficient data"}
        recent = df.tail(25).copy()
        recent["high_20"] = recent["high"].rolling(20).max()
        recent["low_20"] = recent["low"].rolling(20).min()
        last = recent.iloc[-1]
        sweep_high = bool(last["high"] > last["high_20"] and last["close"] < last["high"])
        sweep_low = bool(last["low"] < last["low_20"] and last["close"] > last["low"])
        return {"long": sweep_low, "short": sweep_high, "reason": "sweep_low" if sweep_low else "sweep_high" if sweep_high else "none"}

    def _breakout_state(self, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < 20:
            return {"breakout_long": False, "breakout_short": False}
        last = float(df["close"].iloc[-1])
        high20 = float(df["high"].tail(20).max())
        low20 = float(df["low"].tail(20).min())
        return {"breakout_long": last >= high20, "breakout_short": last <= low20, "high20": high20, "low20": low20}

    def combine_signal(self, df: pd.DataFrame, symbol: str, exchange) -> Optional[Dict[str, Any]]:
        atr_pct = get_atr_pct(df)
        if atr_pct < MIN_ATR_PCT:
            self.rejection_reasons.append("Low ATR / low volatility")
            return None

        imbalance = MarketData(exchange).fetch_order_book_imbalance(symbol)
        funding = MarketData(exchange).fetch_funding_rate(symbol)
        regime = detect_regime(df)
        self.last_regime = regime

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

        volume_spike = compute_volume_spike(df)
        liq_quality = compute_liquidity_quality(df, imbalance, volume_spike)
        if liq_quality < 0.45:
            self.rejection_reasons.append("Poor liquidity quality")
            return None

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
        dominant_strategy = max(strategies, key=lambda x: x[0] * float(weights.get(x[2], 0.1)))[2] if strategies else "ensemble"

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
            "imbalance": float(imbalance),
            "funding": float(funding),
            "atr_pct": float(atr_pct),
            "volume_spike": float(volume_spike),
            "liquidity_quality": float(liq_quality),
        }

    def diagnose_no_setup(self, symbols: List[str], exchange) -> None:
        sample = symbols[0] if symbols else None
        if not sample:
            return
        df = MarketData(exchange).fetch_ohlcv(sample)
        if df is None or df.empty:
            return
        imbalance = MarketData(exchange).fetch_order_book_imbalance(sample)
        atr = get_atr_pct(df)
        vol = compute_volume_spike(df)
        needs = self.needs_analyzer.analyze(df, sample, imbalance, atr, vol)
        reason_counts = Counter(self.rejection_reasons)
        msg = "📉 Market Scan Complete — No Setup Found\n\n"
        msg += "Primary constraints:\n"
        if needs:
            for n in needs[:6]:
                msg += f"• {n}\n"
        else:
            msg += "• No strong confluence\n"
        if reason_counts:
            msg += "\nTop rejection reasons:\n"
            for reason, count in reason_counts.most_common(5):
                msg += f"• {reason}: {count}\n"
        msg += f"\nSystem stats:\nATR={atr:.5f}\nVolume spike={vol:.2f}\nImbalance={imbalance:.2f}\n"
        send_telegram(msg)

# ======================================================
# RISK ENGINE
# ======================================================
class RiskEngine:
    def __init__(self, memory: Memory):
        self.memory = memory

    def compute_position_size(self, equity: float, confidence: float, recent_trades: List[Dict[str, Any]]) -> float:
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

    def calculate_trade_params(self, equity: float, stop_dist_pct: float) -> Dict[str, Any]:
        stop_dist_pct = max(float(stop_dist_pct), MIN_ATR_PCT)
        if equity < BOOTSTRAP_THRESHOLD:
            mode = "BOOTSTRAP"
            size = MIN_NOTIONAL_EXCHANGE
            lev = max(MIN_LEVERAGE, size / max(equity, 0.01))
        else:
            mode = "PROFESSIONAL"
            ideal_size = (equity * RISK_PER_TRADE_PCT) / stop_dist_pct
            size = max(MIN_NOTIONAL_EXCHANGE, ideal_size)
            lev = max(MIN_LEVERAGE, size / max(equity, 0.01))
        lev = clamp(lev, MIN_LEVERAGE, MAX_LEVERAGE)
        size = max(size, MIN_NOTIONAL_EXCHANGE)
        return {"mode": mode, "size": round(size, 4), "lev": round(lev, 2)}

# ======================================================
# EXECUTION ENGINE
# ======================================================
class ExecutionEngine:
    def __init__(self, memory: Memory, exchange):
        self.memory = memory
        self.exchange = exchange

    def attempt_set_leverage(self, symbol: str, leverage: int) -> None:
        try:
            self.exchange.set_margin_mode("isolated", symbol)
        except Exception:
            pass
        try:
            self.exchange.set_leverage(int(leverage), symbol)
        except Exception:
            pass

    def place_live_entry(self, symbol: str, side: str, amount: float, sl_price: float, tp_price: float):
        order_side = "buy" if side == "long" else "sell"
        params = {"stopLossPrice": sl_price, "takeProfitPrice": tp_price}
        return safe_call(lambda: self.exchange.create_order(symbol, "market", order_side, amount, None, params=params))

    def enter_trade(self, signal: Dict[str, Any], df: pd.DataFrame, risk: RiskEngine, trade_lifecycle: TradeLifecycle):
        if LIVE_TRADING and DRY_RUN:
            send_telegram("❌ CONFIG ERROR: both LIVE_TRADING and DRY_RUN enabled")
            return

        symbol = signal["symbol"]
        side = signal["direction"]
        price = float(df["close"].iloc[-1])
        effective_entry = apply_friction(price, is_entry=True)
        stop_dist_pct = estimate_stop_distance_pct(df)
        params = risk.calculate_trade_params(self.memory.get_equity(), stop_dist_pct)
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
            f"🚀 HYBRID AI SIGNAL\n"
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
            f"TP: {tp_price:.6f}\n"
            f"Liquidity Quality: {signal['liquidity_quality']:.2f}"
        )
        send_telegram(msg)

        if DRY_RUN or not LIVE_TRADING:
            trade_lifecycle.memory.data["open_trade"] = {
                "id": make_id(symbol.replace("/", "_")),
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
            trade_lifecycle.memory.save()
            return

        self.attempt_set_leverage(symbol, int(params["lev"]))
        order = self.place_live_entry(symbol, side, amount_base, sl_price, tp_price)
        if not order:
            send_telegram(f"⚠️ Entry failed for {symbol}")
            return
        trade_lifecycle.memory.data["open_trade"] = {
            "id": order.get("id", make_id(symbol.replace("/", "_"))),
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
        trade_lifecycle.memory.data["last_trade_symbol"] = symbol
        trade_lifecycle.memory.data["last_trade_side"] = side
        trade_lifecycle.memory.save()

    def _log_raw_setup(self, signal: Dict[str, Any], params: Dict[str, Any], price: float, stop_dist_pct: float):
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
            "atr_pct": signal.get("atr_pct", 0),
            "volume_spike": signal.get("volume_spike", 0),
            "liquidity_quality": signal.get("liquidity_quality", 0),
        }])
        append_to_csv(self.memory.drive, RAW_SETUPS_LOG, DRIVE_FOLDER_ID, row)

# ======================================================
# OBSERVABILITY
# ======================================================
class Observability:
    def __init__(self, memory: Memory):
        self.memory = memory

    def heartbeat(self, run_count: int, last_regime: str, open_trade_exists: bool, equity: float):
        last_run = self.memory.get_last_run() or "never"
        msg = (
            f"🟢 CozyHybridAI Heartbeat\n"
            f"Time: {datetime.utcnow().isoformat()}\n"
            f"Run count: {run_count}\n"
            f"Last run: {last_run}\n"
            f"Open trade: {open_trade_exists}\n"
            f"Equity: ${equity:.2f}\n"
            f"Regime: {last_regime}\n"
            f"Dry-run: {DRY_RUN} | Live: {LIVE_TRADING}"
        )
        send_telegram(msg)
        self.memory.data["heartbeat"]["last_heartbeat_sent"] = datetime.utcnow().isoformat()
        self.memory.save()

    def market_pulse(self, symbols_scanned: int, best_symbol: Optional[str], best_score: Optional[float]):
        msg = (
            f"📊 Market Pulse\n"
            f"Scanned: {symbols_scanned}\n"
            f"Best symbol: {best_symbol or 'none'}\n"
            f"Best score: {best_score if best_score is not None else 'n/a'}\n"
            f"Equity: ${self.memory.get_equity():.2f}"
        )
        send_telegram(msg)
        self.memory.data["heartbeat"]["last_market_pulse_sent"] = datetime.utcnow().isoformat()
        self.memory.save()

    def stall_watchdog(self):
        last = self.memory.get_last_run()
        if not last:
            return
        try:
            last_dt = datetime.fromisoformat(last)
            diff = (datetime.utcnow() - last_dt).total_seconds() / 60
            if diff > STALL_ALERT_AFTER_MINUTES:
                send_telegram(f"🚨 WARNING: scanner stalled for {diff:.1f} minutes")
        except Exception:
            pass

# ======================================================
# MAIN ENGINE
# ======================================================
class CozyHybridAI:
    def __init__(self):
        self.drive = get_drive_service() if LOG_TO_DRIVE else None
        self.memory = Memory(self.drive, DRIVE_FOLDER_ID)
        self.exchange = ccxt.bitget({
            "apiKey": BITGET_API_KEY,
            "secret": BITGET_SECRET,
            "password": BITGET_PASSWORD,
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
            "timeout": 20000,
        })
        if USE_DEMO_TRADING:
            try:
                self.exchange.enable_demo_trading(True)
            except Exception:
                pass
        try:
            if DRY_RUN:
                self.exchange.set_sandbox_mode(True)
        except Exception:
            pass
        try:
            self.exchange.load_markets()
        except Exception:
            pass

        self.market = MarketData(self.exchange)
        self.signal = SignalEngine(self.memory)
        self.risk = RiskEngine(self.memory)
        self.exec = ExecutionEngine(self.memory, self.exchange)
        self.obs = Observability(self.memory)
        self.lifecycle = TradeLifecycle(self.memory, self.market, self.exchange, self.drive)
        self.run_counter = int(self.memory.data.get("heartbeat", {}).get("run_count", 0))
        self._last_pulse_cycle = 0
        self._last_heartbeat_cycle = 0
        self.last_regime = "unknown"

    def _survival_gate(self) -> bool:
        self.memory.check_reset_window()
        if self.memory.get_consecutive_losses() >= MAX_CONSECUTIVE_LOSSES:
            send_telegram(f"🛑 HALTED: {self.memory.get_consecutive_losses()} consecutive losses")
            return False
        if self.memory.get_daily_loss() >= MAX_DAILY_LOSS_PCT:
            send_telegram(f"⛔ HALTED: daily loss {self.memory.get_daily_loss() * 100:.1f}%")
            return False
        return True

    def _heartbeat_and_watchdog(self):
        self.obs.stall_watchdog()
        if self.run_counter - self._last_heartbeat_cycle >= HEARTBEAT_INTERVAL_RUNS:
            self.obs.heartbeat(self.run_counter, self.last_regime, self.memory.data.get("open_trade") is not None, self.memory.get_equity())
            self._last_heartbeat_cycle = self.run_counter
        if self.run_counter - self._last_pulse_cycle >= MARKET_PULSE_INTERVAL_RUNS:
            best = self.memory.data.get("best_candidate") or {}
            self.obs.market_pulse(MAX_SYMBOLS, best.get("symbol"), best.get("final_score"))
            self._last_pulse_cycle = self.run_counter

    def _update_trade_day(self) -> None:
        today = datetime.now().date().isoformat()
        if self.memory.data.get("last_trade_day") != today:
            self.memory.reset_daily_loss()
            self.memory.data["last_trade_day"] = today
            self.memory.save()

    def run(self) -> None:
        log("CozyHybridAI.run() started")
        send_telegram("🟢 Trading AI started successfully")

        try:
            self.memory.sync_equity_from_exchange(self.exchange)
            self._update_trade_day()
            if not self._survival_gate():
                return

            # Always manage open trade first
            self.lifecycle.manage_open_trade()

            # If a trade is open, do not search for a new one
            if self.memory.data.get("open_trade"):
                self.run_counter = int(self.memory.data["heartbeat"].get("run_count", 0)) + 1
                self.memory.update_heartbeat(self.run_counter)
                self._heartbeat_and_watchdog()
                self.memory.adjust_filters_based_on_performance()
                return

            self.run_counter = int(self.memory.data["heartbeat"].get("run_count", 0)) + 1
            self.memory.update_heartbeat(self.run_counter)

            symbols = self.market.get_top_symbols(MAX_SYMBOLS)
            if not symbols:
                send_telegram("No futures symbols found.")
                return

            best_signal = None
            best_df = None
            self.signal.rejection_reasons = []

            for sym in symbols:
                try:
                    df = self.market.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=OHLCV_LIMIT)
                    if df is None or len(df) < 50:
                        self.signal.rejection_reasons.append("OHLCV data insufficient")
                        continue

                    signal = self.signal.combine_signal(df, sym, self.exchange)
                    if not signal:
                        continue
                    if signal["direction"] == "neutral":
                        self.signal.rejection_reasons.append("Direction neutral")
                        continue

                    threshold = self.memory.get_dynamic_threshold() + signal["bias"]
                    if signal["hour"] in self.memory.data.get("toxic_hours", []):
                        threshold += 0.08

                    if signal["final_score"] < max(threshold, MIN_SIGNAL_SCORE):
                        self.signal.rejection_reasons.append(
                            f"Score below threshold ({signal['final_score']:.2f} < {threshold:.2f})"
                        )
                        continue

                    if best_signal is None or signal["final_score"] > best_signal["final_score"]:
                        best_signal = signal
                        best_df = df
                        self.last_regime = signal["regime"]
                        if signal["final_score"] >= FAST_ENTRY_SCORE:
                            log("Fast-entry confluence hit; stopping scan early.")
                            break
                except Exception as exc:
                    log(f"Error on {sym}: {exc}")
                    self.signal.rejection_reasons.append(f"Exception: {str(exc)[:80]}")

            if not best_signal or best_df is None:
                log("No high-confidence signal.")
                self.memory.data["last_rejection"] = "No high-confidence setup found this cycle."
                self.memory.data["best_candidate"] = None
                self.memory.data["recent_no_setup_reasons"] = self.signal.rejection_reasons[-20:]
                self.memory.save()
                self.signal.diagnose_no_setup(symbols, self.exchange)
                self._heartbeat_and_watchdog()
                self.memory.adjust_filters_based_on_performance()
                return

            self.memory.data["best_candidate"] = best_signal
            self.memory.data["last_rejection"] = None
            self.memory.save()

            self.exec.enter_trade(best_signal, best_df, self.risk, self.lifecycle)
            self.memory.adjust_filters_based_on_performance()

        except Exception as exc:
            send_telegram(f"🔴 Scanner crashed: {exc}")
            log(traceback.format_exc())
            raise

# ======================================================
# ENTRY POINT (with infinite loop for 24/7 operation)
# ======================================================
if __name__ == "__main__":
    print("DEBUG: Before creating AI instance")
    while True:
        try:
            ai = CozyHybridAI()
            print("DEBUG: AI instance created, running")
            ai.run()
            print("DEBUG: Run finished, sleeping 60 seconds")
            time.sleep(60)
        except KeyboardInterrupt:
            print("Shutdown requested")
            break
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            time.sleep(30)
