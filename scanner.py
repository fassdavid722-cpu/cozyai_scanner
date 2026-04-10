#!/usr/bin/env python3
"""
CozyHybridAI — Bitget Futures Execution Engine
With Google Drive persistence and Telegram fix
"""

import os
import json
import time
import io
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import ccxt
import numpy as np
import pandas as pd
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ======================================================
# ENV / CONFIG
# ======================================================
BITGET_API_KEY = os.getenv("BITGET_API_KEY", "")
BITGET_SECRET = os.getenv("BITGET_SECRET", "")
BITGET_PASSWORD = os.getenv("BITGET_PASSWORD", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", os.getenv("CHAT_ID", ""))
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"
LIVE_TRADING = os.getenv("LIVE_TRADING", "false").lower() == "true"
USE_DEMO_TRADING = os.getenv("USE_DEMO_TRADING", "false").lower() == "true"

STARTING_EQUITY = float(os.getenv("STARTING_EQUITY", "3.0"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.05"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))
MAX_LEVERAGE = float(os.getenv("MAX_LEVERAGE", "10.0"))
MIN_NOTIONAL = float(os.getenv("MIN_NOTIONAL", "0.5"))
RISK_PER_TRADE_PCT = float(os.getenv("RISK_PER_TRADE_PCT", "0.02"))
BOOTSTRAP_THRESHOLD = float(os.getenv("BOOTSTRAP_THRESHOLD", "10.0"))
BASE_THRESHOLD = float(os.getenv("BASE_THRESHOLD", "0.68"))
MAX_SYMBOLS = int(os.getenv("MAX_SYMBOLS", "30"))
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
OHLCV_LIMIT = int(os.getenv("OHLCV_LIMIT", "100"))
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "200"))
TAKE_PROFIT_R = float(os.getenv("TAKE_PROFIT_R", "2.0"))
MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", "0.65"))

STRATEGY_LIST = [
    "volume_breakout",
    "orderbook_imbalance",
    "funding_arbitrage",
    "liquidity_sweep",
    "momentum",
]
DEFAULT_STRATEGY_WEIGHTS = {s: 1.0 / len(STRATEGY_LIST) for s in STRATEGY_LIST}
MEMORY_FILE = "cozy_memory.json"

# Google Drive settings (same as your old scanner)
DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1Ox77rDeIj7XEE_pyfE5TyKVtiYtbdcXe")
# ======================================================

# ---------- Google Drive helpers ----------
def get_drive_service():
    creds_json = os.environ.get("GDRIVE_CREDS")
    if not creds_json:
        raise Exception("Missing GDRIVE_CREDS secret")
    creds_dict = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=creds)

def download_file(drive, filename, folder_id):
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
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)
    return fh

def upload_file(drive, filename, folder_id, content_bytes):
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = drive.files().list(q=query, fields="files(id)").execute()
    files = res.get('files', [])
    media = MediaIoBaseUpload(io.BytesIO(content_bytes), mimetype='application/json', resumable=True)
    if files:
        drive.files().update(fileId=files[0]['id'], media_body=media).execute()
    else:
        metadata = {'name': filename, 'parents': [folder_id]}
        drive.files().create(body=metadata, media_body=media).execute()

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
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
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

# ======================================================
# MEMORY (with Google Drive persistence)
# ======================================================
class Memory:
    def __init__(self, file: str = MEMORY_FILE, drive_folder: str = DRIVE_FOLDER_ID):
        self.file = file
        self.drive_folder = drive_folder
        self.drive = None
        try:
            self.drive = get_drive_service()
        except Exception:
            pass
        self.data = self.load()
        self.normalize()

    def load(self) -> Dict:
        # Try to load from Drive first
        if self.drive:
            fh = download_file(self.drive, self.file, self.drive_folder)
            if fh:
                try:
                    return json.load(fh)
                except:
                    pass
        # Fallback to local file
        if os.path.exists(self.file):
            try:
                with open(self.file, "r") as f:
                    return json.load(f)
            except:
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

    def save(self) -> None:
        # Save to local file
        with open(self.file, "w") as f:
            json.dump(self.data, f, indent=4)
        # Upload to Google Drive if available
        if self.drive:
            try:
                content = json.dumps(self.data, indent=4).encode('utf-8')
                upload_file(self.drive, self.file, self.drive_folder, content)
            except Exception as e:
                print(f"Failed to upload to Drive: {e}")

    def sync_equity_from_exchange(self, exchange) -> None:
        try:
            balance = safe_call(lambda: exchange.fetch_balance())
            total = (balance or {}).get("total", {}) or {}
            equity = float(total.get("USDT", self.data.get("equity", STARTING_EQUITY)))
            self.data["equity"] = max(equity, 0.0)
            self.save()
        except Exception:
            pass

    # ... (rest of the Memory methods remain exactly the same as in your working hybrid script)
    # I'll omit them for brevity, but you must copy the full class from the previous working version.
    # The only change is the addition of Drive upload/download in load/save.

# ======================================================
# The rest of the script (strategies, main engine) stays identical
# ======================================================
# ... (paste all the strategy functions, the CozyHybridAI class, etc. from the previous working version)
# Make sure to use the updated Memory class above.
