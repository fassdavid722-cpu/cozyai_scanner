#!/usr/bin/env python3
# CozyAI – AI‑Driven Multi‑Strategy Scanner
# The AI looks at market numbers, chooses strategy, predicts EV.

import ccxt
import pandas as pd
import numpy as np
import time
import os
import json
import requests
import xgboost as xgb
import tempfile
from datetime import datetime, date
from io import BytesIO
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# ======================================================
# CONFIGURATION
# ======================================================
DRIVE_FOLDER_ID = '1Ox77rDeIj7XEE_pyfE5TyKVtiYtbdcXe'
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID = os.environ.get('CHAT_ID', '')
MAX_SYMBOLS = 30
AUTO_TRADE = False
STARTING_EQUITY = 3.0
MAX_DAILY_LOSS_PCT = 0.05
MAX_CONSECUTIVE_LOSSES = 3
EV_THRESHOLD = 0.5          # Only trade if predicted EV > 0.5
# ======================================================

# ---------- Google Drive helpers (unchanged) ----------
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

def read_csv(drive, filename, folder_id):
    fh = download_file(drive, filename, folder_id)
    if fh:
        return pd.read_csv(fh)
    return None

# ---------- Survival & risk (same as before) ----------
def get_survival_status(drive, folder_id):
    df = read_csv(drive, 'trade_log.csv', folder_id)
    if df is None or df.empty:
        return {'daily_loss_pct': 0.0, 'consecutive_losses': 0, 'equity': STARTING_EQUITY}
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    today = date.today()
    today_trades = df[df['timestamp'].dt.date == today]
    daily_loss_pct = abs(today_trades[today_trades['pnl'] < 0]['pnl'].sum()) / STARTING_EQUITY if STARTING_EQUITY > 0 else 0
    df_sorted = df.sort_values('timestamp', ascending=False)
    consecutive_losses = 0
    for _, row in df_sorted.iterrows():
        if row['pnl'] < 0:
            consecutive_losses += 1
        else:
            break
    total_pnl = df['pnl'].sum()
    equity = STARTING_EQUITY + total_pnl
    return {'daily_loss_pct': daily_loss_pct, 'consecutive_losses': consecutive_losses, 'equity': max(equity, 0.5)}

def check_survival_conditions(status, predicted_ev):
    if status['daily_loss_pct'] >= MAX_DAILY_LOSS_PCT:
        return False, f"Daily loss limit reached ({status['daily_loss_pct']*100:.1f}%)"
    if status['consecutive_losses'] >= MAX_CONSECUTIVE_LOSSES:
        return False, f"Consecutive losses ({status['consecutive_losses']})"
    if predicted_ev < EV_THRESHOLD:
        return False, f"Predicted EV too low ({predicted_ev:.2f})"
    return True, "OK"

# ---------- Feature extraction (the AI's "eyes") ----------
def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0

def compute_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if not adx.empty else 20

def compute_range_efficiency(df, lookback=20):
    if len(df) < lookback:
        return 0.0
    close = df['close'].tail(lookback)
    net_move = abs(close.iloc[-1] - close.iloc[0])
    total_move = close.diff().abs().sum()
    if total_move == 0:
        return 0.0
    return float(net_move / total_move)

def compute_volume_spike(df, lookback=20):
    if len(df) < lookback:
        return 1.0
    avg_vol = df['volume'].rolling(lookback).mean().iloc[-1]
    last_vol = df['volume'].iloc[-1]
    return last_vol / avg_vol if avg_vol > 0 else 1.0

def compute_ob_imbalance(df):
    last = df.iloc[-1]
    vol_spike = compute_volume_spike(df)
    if last['close'] > last['open'] and vol_spike > 1.2:
        return 0.6
    elif last['close'] < last['open'] and vol_spike > 1.2:
        return -0.6
    else:
        return 0.0

def compute_regime_features(df):
    adx = compute_adx(df)
    eff = compute_range_efficiency(df)
    atr = compute_atr(df)
    price = df['close'].iloc[-1]
    vol = atr / price if price > 0 else 0
    # Encode regime as numeric for the model (optional, but we can include raw adx, eff, vol)
    return {
        'adx': adx,
        'efficiency': eff,
        'volatility': vol,
        'trending': 1 if adx > 25 and eff > 0.4 else 0,
        'ranging': 1 if adx < 20 and eff < 0.3 else 0,
        'volatile': 1 if vol > 0.005 else 0,
        'choppy': 1 if not (adx > 25 and eff > 0.4) and not (adx < 20 and eff < 0.3) and vol <= 0.005 else 0
    }

# ---------- Strategy detectors (generate candidate setups) ----------
def detect_liquidity_sweep_fvg(df):
    # Your original detection – simplified version, you can paste your full code
    # Returns dict with 'type' (long/short), 'entry', 'sl', 'tp', and a placeholder 'confidence' (will be overridden by model)
    # For brevity, I'll include a minimal version. Replace with your actual functions.
    # ... (keep your existing detect_liquidity_sweep and detect_fvg functions)
    # For now, return None to avoid errors; you will replace with your code.
    return None

def detect_ema_pullback(df):
    # ... (your existing EMA pullback detection)
    return None

def detect_volume_breakout(df):
    # ... (your existing volume breakout detection)
    return None

# ---------- Main scanner with AI model ----------
def main():
    print(f"Starting AI‑Driven Multi‑Strategy Scanner at {datetime.utcnow()}")
    drive = get_drive_service()
    status = get_survival_status(drive, DRIVE_FOLDER_ID)
    print(f"Equity: ${status['equity']:.2f}, Daily loss: {status['daily_loss_pct']*100:.2f}%, Consecutive losses: {status['consecutive_losses']}")

    # Load AI model
    model_file = download_file(drive, 'cozyai_model.json', DRIVE_FOLDER_ID)
    model = None
    if model_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            tmp.write(model_file.getvalue())
            tmp_path = tmp.name
        model = xgb.Booster()
        model.load_model(tmp_path)
        os.unlink(tmp_path)
        print("AI model loaded.")
    else:
        print("No AI model – falling back to rule‑based confidence.")

    # Get top symbols (you can use hot token ranking here)
    exchange = ccxt.bitget()
    tickers = exchange.fetch_tickers()
    symbols = [s for s in tickers if s.endswith('/USDT')][:MAX_SYMBOLS]  # simplified
    setups = []

    for sym in symbols:
        try:
            df = fetch_candles(sym, limit=100, timeframe='15m')
            if df is None or len(df) < 60:
                continue

            # Extract all market features (the AI's "view")
            regime = compute_regime_features(df)
            volume_spike = compute_volume_spike(df)
            pressure = compute_ob_imbalance(df)
            atr = compute_atr(df)
            adx = regime['adx']
            efficiency = regime['efficiency']
            volatility = regime['volatility']

            # Candidate strategies (you can add more)
            strategies = []
            cand1 = detect_liquidity_sweep_fvg(df)
            if cand1:
                strategies.append(('LiquiditySweep+FVG', cand1))
            cand2 = detect_ema_pullback(df)
            if cand2:
                strategies.append(('EMAPullback', cand2))
            cand3 = detect_volume_breakout(df)
            if cand3:
                strategies.append(('VolumeBreakout', cand3))

            if not strategies:
                continue

            # For each candidate, assemble feature vector and ask the model for EV
            best_candidate = None
            best_ev = -1
            for strat_name, cand in strategies:
                # Feature vector (must match training order)
                features = [
                    volume_spike,
                    adx,
                    efficiency,
                    volatility,
                    pressure,
                    atr,
                    regime['trending'],
                    regime['ranging'],
                    regime['volatile'],
                    regime['choppy'],
                    # Add strategy type as one-hot? Better to let model learn from market features alone.
                    # We'll include strategy type as a feature (encoded 0/1/2)
                    0 if strat_name == 'LiquiditySweep+FVG' else (1 if strat_name == 'EMAPullback' else 2)
                ]
                if model:
                    dmat = xgb.DMatrix([features])
                    ev = model.predict(dmat)[0]
                else:
                    # Fallback rule‑based confidence (as before)
                    ev = cand.get('confidence', 0.5)
                if ev > best_ev:
                    best_ev = ev
                    best_candidate = cand
                    best_strategy = strat_name

            if best_candidate is None or best_ev < EV_THRESHOLD:
                continue

            # Apply survival and pressure filters (optional)
            if best_candidate['type'] == 'long' and pressure < 0:
                continue
            if best_candidate['type'] == 'short' and pressure > 0:
                continue

            setups.append({
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': sym,
                'direction': best_candidate['type'].upper(),
                'entry': best_candidate['entry'],
                'sl': best_candidate['sl'],
                'tp': best_candidate['tp'],
                'pred_ev': best_ev,
                'strategy': best_strategy,
                'volume_spike': volume_spike,
                'adx': adx,
                'efficiency': efficiency,
                'volatility': volatility,
                'pressure': pressure
            })
        except Exception as e:
            print(f"Error on {sym}: {e}")
        time.sleep(0.5)

    if setups:
        df_setups = pd.DataFrame(setups)
        df_setups = df_setups.sort_values('pred_ev', ascending=False)
        append_to_csv(drive, 'raw_setups.csv', DRIVE_FOLDER_ID, df_setups)
        best = df_setups.iloc[0]
        ok, reason = check_survival_conditions(status, best['pred_ev'])
        if not ok:
            print(f"Trade blocked: {reason}")
        else:
            msg = (f"🦢 AI‑DRIVEN SIGNAL\n"
                   f"Symbol: {best['symbol']}\n"
                   f"Direction: {best['direction']}\n"
                   f"Predicted EV: {best['pred_ev']:.2f}R\n"
                   f"Strategy: {best['strategy']}\n"
                   f"Entry: {best['entry']:.4f}\n"
                   f"SL: {best['sl']:.4f}\n"
                   f"TP: {best['tp']:.4f}\n"
                   f"Vol spike: {best['volume_spike']:.2f} | ADX: {best['adx']:.1f} | Pressure: {best['pressure']:.2f}")
            send_telegram(msg)
            print(f"Found {len(setups)} setups, alerted top.")
            if AUTO_TRADE:
                # Place order using best['entry'], best['sl'], best['tp'], and position sizing based on best['pred_ev']
                print("Auto‑trade would execute.")
    else:
        print("No setups found.")

def fetch_candles(symbol, limit=100, timeframe='15m'):
    exchange = ccxt.bitget({'enableRateLimit': True})
    try:
        candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(candles, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Fetch error {symbol}: {e}")
        return None

def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

if __name__ == "__main__":
    main()
