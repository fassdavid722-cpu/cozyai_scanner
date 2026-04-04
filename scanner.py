#!/usr/bin/env python3
# CozyAI Scanner – AI‑Powered (uses XGBoost model from Drive)

import ccxt
import pandas as pd
import numpy as np
import time
import os
import json
import requests
import xgboost as xgb
from datetime import datetime
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ======================================================
# CONFIGURATION – EDIT THESE
# ======================================================
DRIVE_FOLDER_ID = '1Ox77rDeIj7XEE_pyfE5TyKVtiYtbdcXe'   # your folder ID
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')
MAX_SYMBOLS = 30
# ======================================================

# ---------- Google Drive helpers (same as before) ----------
def get_drive_service():
    creds_json = os.environ.get('GDRIVE_CREDS')
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
    # Get file's MIME type
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

def upload_file(drive, filename, folder_id, content_bytes):
    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    res = drive.files().list(q=query, fields="files(id)").execute()
    files = res.get('files', [])
    media = MediaIoBaseUpload(BytesIO(content_bytes), mimetype='text/csv', resumable=True)
    if files:
        drive.files().update(fileId=files[0]['id'], media_body=media).execute()
    else:
        metadata = {'name': filename, 'parents': [folder_id]}
        drive.files().create(body=metadata, media_body=media).execute()

def append_to_csv(drive, filename, folder_id, new_rows_df):
    existing = download_file(drive, filename, folder_id)
    if existing:
        old_df = pd.read_csv(existing)
        df = pd.concat([old_df, new_rows_df], ignore_index=True)
    else:
        df = new_rows_df
    buf = BytesIO()
    df.to_csv(buf, index=False)
    upload_file(drive, filename, folder_id, buf.getvalue())

def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

# ---------- Market data and feature extraction ----------
def get_top_symbols(limit=MAX_SYMBOLS):
    exchange = ccxt.bitget()
    tickers = exchange.fetch_tickers()
    pairs = []
    for symbol, data in tickers.items():
        if symbol.endswith('/USDT') and 'quoteVolume' in data and data['quoteVolume']:
            pairs.append((symbol, data['quoteVolume']))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:limit]]

def fetch_candles(symbol, limit=150):
    exchange = ccxt.bitget({'enableRateLimit': True})
    candles = exchange.fetch_ohlcv(symbol, '1m', limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def detect_liquidity_sweep(df, lookback=20):
    if len(df) < lookback+5:
        return None
    recent = df.tail(30).copy()
    recent['high_20'] = recent['high'].rolling(lookback).max()
    recent['low_20'] = recent['low'].rolling(lookback).min()
    recent['sweep_high'] = (recent['high'] > recent['high_20'].shift(1)) & (recent['close'] < recent['high'])
    recent['sweep_low'] = (recent['low'] < recent['low_20'].shift(1)) & (recent['close'] > recent['low'])
    sweep = recent[recent['sweep_high'] | recent['sweep_low']]
    if not sweep.empty:
        last = sweep.iloc[-1]
        direction = 'long' if last['sweep_low'] else 'short'
        range_high = last['high_20']
        range_low = last['low_20']
        if direction == 'long':
            sweep_size = (range_low - last['low']) / (range_high - range_low) if (range_high - range_low) > 0 else 0
        else:
            sweep_size = (last['high'] - range_high) / (range_high - range_low) if (range_high - range_low) > 0 else 0
        return {
            'index': last.name,
            'direction': direction,
            'sweep_size': abs(sweep_size),
            'price': last['close']
        }
    return None

def detect_fvg(df, sweep_idx):
    if sweep_idx+3 >= len(df):
        return None
    c1 = df.iloc[sweep_idx]
    c2 = df.iloc[sweep_idx+1]
    c3 = df.iloc[sweep_idx+2]
    if c1['high'] < c3['low']:
        return {'type': 'bullish', 'high': c1['high'], 'low': c3['low']}
    elif c1['low'] > c3['high']:
        return {'type': 'bearish', 'high': c3['high'], 'low': c1['low']}
    return None

def compute_volume_spike(df):
    if len(df) < 20:
        return 1.0
    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
    last_vol = df['volume'].iloc[-1]
    return last_vol / avg_vol if avg_vol > 0 else 1.0

def compute_displacement_strength(df, idx):
    if idx >= len(df):
        return 0
    candle = df.iloc[idx]
    body = abs(candle['close'] - candle['open'])
    wick = max(candle['high'] - max(candle['close'], candle['open']), min(candle['close'], candle['open']) - candle['low'])
    if wick == 0:
        return body / (body + 1e-6)
    return body / (body + wick)

def get_session():
    hour = datetime.utcnow().hour
    if 7 <= hour < 16:
        return 'London'
    elif 12 <= hour < 21:
        return 'NewYork'
    else:
        return 'Asia'

def compute_regime(df):
    atr = df['close'].rolling(14).apply(lambda x: np.ptp(x) if len(x) == 14 else np.nan).iloc[-1]
    price = df['close'].iloc[-1]
    vol = atr / price if price > 0 else 0
    x = np.arange(20)
    y = df['close'].tail(20).values
    slope = np.polyfit(x, y, 1)[0] if len(y) == 20 else 0
    return {'volatility': vol, 'trend_slope': slope}

def fetch_funding_and_oi(symbol):
    exchange = ccxt.bitget({'options': {'defaultType': 'swap'}})
    try:
        base = symbol.split('/')[0]
        futures_sym = f"{base}USDT:USDT"
        funding = exchange.fetch_funding_rate(futures_sym)
        oi = exchange.fetch_open_interest(futures_sym)
        return {
            'funding_rate': funding.get('fundingRate', 0),
            'open_interest': oi.get('openInterest', 0)
        }
    except:
        return {'funding_rate': 0, 'open_interest': 0}

# ---------- Main scanning function with AI model ----------
def main():
    print(f"Starting AI scanner at {datetime.utcnow()}")
    drive = get_drive_service()

    # Download AI model from Drive
    model_file = download_file(drive, 'cozyai_model.json', DRIVE_FOLDER_ID)
    model = None
    if model_file:
        # Load XGBoost model
        model = xgb.Booster()
        model.load_model(model_file)
        print("AI model loaded.")
    else:
        print("No model found – using rule‑based fallback.")

    symbols = get_top_symbols(MAX_SYMBOLS)
    print(f"Scanning {len(symbols)} symbols...")

    setups = []
    for sym in symbols:
        try:
            df = fetch_candles(sym)
            if df is None or len(df) < 50:
                continue
            sweep = detect_liquidity_sweep(df)
            if not sweep:
                continue
            fvg = detect_fvg(df, sweep['index'])
            if not fvg:
                continue
            volume_spike = compute_volume_spike(df)
            displacement = compute_displacement_strength(df, sweep['index'])
            funding_oi = fetch_funding_and_oi(sym)
            session = get_session()
            regime = compute_regime(df)

            # Prepare feature vector (must match training order)
            features = [
                sweep['sweep_size'],
                fvg['width'] if 'width' in fvg else 0.0,
                volume_spike,
                displacement,
                funding_oi['funding_rate'],
                funding_oi['open_interest'],
                regime['volatility'],
                regime['trend_slope']
            ]

            if model:
                # Predict expected R-multiple
                dmatrix = xgb.DMatrix([features])
                pred_ev = model.predict(dmatrix)[0]
            else:
                # Rule-based fallback (0-1 scale)
                pred_ev = (sweep['sweep_size'] * 0.4 + volume_spike * 0.3 + displacement * 0.3)

            setups.append({
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': sym,
                'direction': 'LONG' if sweep['direction'] == 'long' else 'SHORT',
                'price': sweep['price'],
                'sweep_size': sweep['sweep_size'],
                'fvg_width': fvg['width'] if 'width' in fvg else 0.0,
                'volume_spike': volume_spike,
                'displacement': displacement,
                'funding_rate': funding_oi['funding_rate'],
                'open_interest': funding_oi['open_interest'],
                'session': session,
                'volatility': regime['volatility'],
                'trend_slope': regime['trend_slope'],
                'entry_mid': (fvg['high'] + fvg['low']) / 2,
                'fvg_high': fvg['high'],
                'fvg_low': fvg['low'],
                'pred_ev': pred_ev
            })
        except Exception as e:
            print(f"Error on {sym}: {e}")
        time.sleep(0.5)

    if setups:
        df_setups = pd.DataFrame(setups)
        df_setups = df_setups.sort_values('pred_ev', ascending=False)
        # Log to Drive
        append_to_csv(drive, 'raw_setups.csv', DRIVE_FOLDER_ID, df_setups)
        # Alert top setup
        best = df_setups.iloc[0]
        msg = (f"🤖 COZYAI AI SIGNAL\n"
               f"Symbol: {best['symbol']}\n"
               f"Direction: {best['direction']}\n"
               f"Predicted EV: {best['pred_ev']:.2f}R\n"
               f"Price: {best['price']:.2f}\n"
               f"Entry zone: {best['fvg_low']:.2f}–{best['fvg_high']:.2f}")
        send_telegram(msg)
        print(f"Found {len(setups)} setups, alerted top.")
    else:
        print("No setups found.")

if __name__ == "__main__":
    main()
