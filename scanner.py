#!/usr/bin/env python3
# Simplified CozyAI Scanner – logs setups and sends alerts

import ccxt
import pandas as pd
import numpy as np
import time
import os
import json
import requests
from datetime import datetime
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ======================================================
# CONFIGURATION – EDIT THIS LINE
# ======================================================
DRIVE_FOLDER_ID = '1Ox77rDeIj7XEE_pyfE5TyKVtiYtbdcXe'
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')
MAX_SYMBOLS = 30   # start with 30 symbols
# ======================================================

# Helper: Google Drive
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
    # Get the file's MIME type
    file_meta = drive.files().get(fileId=file_id, fields="mimeType").execute()
    mime_type = file_meta.get('mimeType')
    # If it's a Google Sheets file, export as CSV
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

# Telegram
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

# Market data
def get_top_symbols(limit=MAX_SYMBOLS):
    exchange = ccxt.bitget()
    tickers = exchange.fetch_tickers()
    pairs = []
    for symbol, data in tickers.items():
        if symbol.endswith('/USDT') and 'quoteVolume' in data and data['quoteVolume']:
            pairs.append((symbol, data['quoteVolume']))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pairs[:limit]]

def fetch_candles(symbol, limit=100):
    exchange = ccxt.bitget({'enableRateLimit': True})
    candles = exchange.fetch_ohlcv(symbol, '1m', limit=limit)
    df = pd.DataFrame(candles, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Setup detection
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

# Main
def main():
    print(f"Scanning {len(symbols)} symbols at {datetime.utcnow()}")
    print("Starting scanner...")
    drive = get_drive_service()
    symbols = get_top_symbols(MAX_SYMBOLS)
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
            session = get_session()
            # Simple score (placeholder)
            score = (sweep['sweep_size'] * 0.4 + volume_spike * 0.3 + displacement * 0.3)
            setups.append({
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': sym,
                'direction': 'LONG' if sweep['direction'] == 'long' else 'SHORT',
                'price': sweep['price'],
                'sweep_size': sweep['sweep_size'],
                'fvg_width': 0.0,
                'volume_spike': volume_spike,
                'displacement': displacement,
                'funding_rate': 0.0,
                'open_interest': 0,
                'session': session,
                'volatility': 0.0,
                'trend_slope': 0.0,
                'entry_mid': (fvg['high'] + fvg['low'])/2,
                'fvg_high': fvg['high'],
                'fvg_low': fvg['low'],
                'pred_ev': score
            })
        except Exception as e:
            print(f"Error on {sym}: {e}")
        time.sleep(0.5)
    if setups:
        df_setups = pd.DataFrame(setups)
        append_to_csv(drive, 'raw_setups.csv', DRIVE_FOLDER_ID, df_setups)
        best = df_setups.loc[df_setups['pred_ev'].idxmax()]
        msg = (f"🤖 COZYAI SETUP\nSymbol: {best['symbol']}\nDirection: {best['direction']}\nScore: {best['pred_ev']:.2f}\nPrice: {best['price']:.2f}\nEntry zone: {best['fvg_low']:.2f}–{best['fvg_high']:.2f}")
        send_telegram(msg)
        print(f"Found {len(setups)} setups, alerted.")
    else:
        print("No setups.")

if __name__ == "__main__":
    main()
