#!/usr/bin/env python3
# CozyAI Scanner – God Tier (Full Intelligence + Self-Learning + Monte Carlo)
# Runs on GitHub Actions every 10 minutes

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

# Survival & Risk
MAX_DAILY_LOSS_PCT = 0.05
MAX_CONSECUTIVE_LOSSES = 3
EV_THRESHOLD = 0.5
MIN_VOLUME_SPIKE = 1.2
STARTING_EQUITY = 3.0
AUTO_TRADE = False

# ======================================================
# Google Drive helpers
# ======================================================
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

# ======================================================
# Survival & Risk Engine
# ======================================================
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

def check_survival_conditions(status, predicted_ev, volume_spike):
    if status['daily_loss_pct'] >= MAX_DAILY_LOSS_PCT:
        return False, f"Daily loss limit reached ({status['daily_loss_pct']*100:.1f}%)"
    if status['consecutive_losses'] >= MAX_CONSECUTIVE_LOSSES:
        return False, f"Consecutive losses ({status['consecutive_losses']})"
    if predicted_ev < EV_THRESHOLD:
        return False, f"Predicted EV too low ({predicted_ev:.2f})"
    if volume_spike < MIN_VOLUME_SPIKE:
        return False, f"Volume spike too low ({volume_spike:.2f})"
    return True, "OK"

def compute_adaptive_size(equity, pred_ev, vol, consecutive_losses):
    base_risk = 0.01
    risk = base_risk * (1 + max(pred_ev - 1, 0)) / (1 + vol)
    risk /= (1 + consecutive_losses)
    size_usdt = min(risk * equity, equity * 0.5)
    return max(size_usdt, 0.01)

# ======================================================
# Indicators & Math Intelligence
# ======================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi.iloc[-1] if not rsi.empty else 50

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd.iloc[-1] if not macd.empty else 0, sig.iloc[-1] if not sig.empty else 0

def compute_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    return float(atr) if not np.isnan(atr) else 0.0

def compute_zscore(series, window=20):
    mean = series.rolling(window).mean().iloc[-1]
    std = series.rolling(window).std().iloc[-1]
    if std == 0 or np.isnan(std):
        return 0.0
    return float((series.iloc[-1] - mean) / std)

def compute_momentum_acceleration(df, lookback=5):
    if len(df) < lookback + 2:
        return 0.0
    returns = df['close'].pct_change().tail(lookback)
    acceleration = returns.diff().mean()
    return float(acceleration) if not np.isnan(acceleration) else 0.0

def compute_range_efficiency(df, lookback=20):
    if len(df) < lookback:
        return 0.0
    close = df['close'].tail(lookback)
    net_move = abs(close.iloc[-1] - close.iloc[0])
    total_move = close.diff().abs().sum()
    if total_move == 0:
        return 0.0
    return float(net_move / total_move)

# ======================================================
# Monte Carlo Simulation
# ======================================================
def monte_carlo_outcome(price, pred_ev, vol, num_sim=100):
    simulated = [price + np.random.normal(loc=pred_ev * price, scale=vol * price) for _ in range(num_sim)]
    mean_move = np.mean(simulated) - price
    std_move = np.std(simulated)
    return mean_move, std_move

# ======================================================
# Market Data & Signal Detection
# ======================================================
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
    if len(df) < lookback + 5:
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
    if sweep_idx + 3 >= len(df):
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
    atr = compute_atr(df)
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

# ======================================================
# Confidence & Risk
# ======================================================
def compute_confidence_score(pred_ev, volume_spike, displacement, regime):
    score = 0.0
    score += min(pred_ev / 2, 1.0) * 0.35
    score += min(volume_spike / 3, 1.0) * 0.20
    score += displacement * 0.20
    score += min(regime['volatility'] * 50, 1.0) * 0.10
    score += min(abs(regime['trend_slope']) / 10, 1.0) * 0.15
    return round(score, 4)

def dynamic_risk_size(equity, confidence_score):
    if confidence_score > 0.85:
        risk_pct = 0.04
    elif confidence_score > 0.70:
        risk_pct = 0.03
    elif confidence_score > 0.55:
        risk_pct = 0.02
    else:
        risk_pct = 0.01
    risk_amount = equity * risk_pct
    return {'risk_pct': risk_pct, 'risk_amount': round(risk_amount, 4)}

def get_stop_multiplier(equity):
    if equity < 10:
        return 1.0
    elif equity < 50:
        return 1.2
    elif equity < 200:
        return 1.5
    else:
        return 2.0

# ======================================================
# Telegram & Order Execution
# ======================================================
def send_telegram(message):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def place_order(symbol, direction, entry_price, position_size):
    exchange = ccxt.bitget({
        'apiKey': os.environ.get('BITGET_API_KEY', ''),
        'secret': os.environ.get('BITGET_SECRET_KEY', ''),
        'password': os.environ.get('BITGET_PASSPHRASE', ''),
        'options': {'defaultType': 'spot'}
    })
    try:
        order = exchange.create_order(symbol, 'market', direction.lower(), position_size, None)
        print(f"Order placed: {order}")
        return order
    except Exception as e:
        print(f"Order error: {e}")
        return None

# ======================================================
# Self-Learning Retraining (can be called externally)
# ======================================================
def retrain_model(drive, folder_id, model_path='cozyai_model.json'):
    trades = read_csv(drive, 'trade_log.csv', folder_id)
    if trades is None or len(trades) < 50:
        return
    # Assume we have features in trade_log (you'd need to store them)
    # For simplicity, we'll skip full implementation here.
    print("Retraining not implemented yet – will use weekly workflow.")
    pass

# ======================================================
# Main Scanner Loop
# ======================================================
def main():
    print(f"Starting God Tier CozyAI scanner at {datetime.utcnow()}")
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
        print("No model found – using rule‑based fallback.")

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
                # Add this after compute_atr
def compute_adx(df, period=14):
    """Average Directional Index – values >25 indicate trending."""
    high = df['high']
    low = df['low']
    close = df['close']
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high - low, 
                    abs(high - close.shift(1)), 
                    abs(low - close.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (abs(minus_dm).rolling(period).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx.iloc[-1] if not adx.empty else 20

# Add these filters inside the setup loop (after computing features)
adx = compute_adx(df)
if adx < 25:
    continue   # skip – not trending

volatility_filter = regime['volatility']  # from compute_regime
if volatility_filter < 0.001:   # too low volatility
    continue

efficiency = compute_range_efficiency(df)  # already defined
if efficiency < 0.3:   # choppy
    continue

            # Basic features
            volume_spike = compute_volume_spike(df)
            displacement = compute_displacement_strength(df, sweep['index'])
            funding_oi = fetch_funding_and_oi(sym)
            session = get_session()
            regime = compute_regime(df)

            # Advanced features
            atr = compute_atr(df)
            zscore = compute_zscore(df['close'])
            acceleration = compute_momentum_acceleration(df)
            efficiency = compute_range_efficiency(df)
            rsi = compute_rsi(df['close'])
            macd_val, macd_signal = compute_macd(df['close'])

            # Feature vector for model
            features = [
                sweep['sweep_size'],
                volume_spike,
                displacement,
                funding_oi['funding_rate'],
                funding_oi['open_interest'],
                regime['volatility'],
                regime['trend_slope'],
                atr,
                zscore,
                acceleration,
                efficiency,
                rsi,
                macd_val,
                macd_signal
            ]

            if model:
                dmatrix = xgb.DMatrix([features])
                pred_ev = model.predict(dmatrix)[0]
            else:
                # Fallback rule‑based score
                pred_ev = (sweep['sweep_size'] * 0.4 + volume_spike * 0.3 + displacement * 0.3)

            confidence = compute_confidence_score(pred_ev, volume_spike, displacement, regime)
            risk = dynamic_risk_size(status['equity'], confidence)

            # Monte Carlo simulation for expected move
            mean_move, std_move = monte_carlo_outcome(sweep['price'], pred_ev, regime['volatility'])

            setups.append({
                'timestamp': datetime.utcnow().isoformat(),
                'symbol': sym,
                'direction': 'LONG' if sweep['direction'] == 'long' else 'SHORT',
                'price': sweep['price'],
                'sweep_size': sweep['sweep_size'],
                'volume_spike': volume_spike,
                'displacement': displacement,
                'funding_rate': funding_oi['funding_rate'],
                'open_interest': funding_oi['open_interest'],
                'session': session,
                'volatility': regime['volatility'],
                'trend_slope': regime['trend_slope'],
                'atr': atr,
                'zscore': zscore,
                'acceleration': acceleration,
                'efficiency': efficiency,
                'rsi': rsi,
                'macd': macd_val,
                'macd_signal': macd_signal,
                'pred_ev': pred_ev,
                'confidence': confidence,
                'risk_pct': risk['risk_pct'],
                'risk_amount': risk['risk_amount'],
                'monte_carlo_mean': mean_move,
                'monte_carlo_std': std_move,
                'entry_mid': (fvg['high'] + fvg['low']) / 2,
                'fvg_high': fvg['high'],
                'fvg_low': fvg['low']
            })
        except Exception as e:
            print(f"Error on {sym}: {e}")
        time.sleep(0.5)

    if setups:
        df_setups = pd.DataFrame(setups)
        df_setups = df_setups.sort_values('pred_ev', ascending=False)
        append_to_csv(drive, 'raw_setups.csv', DRIVE_FOLDER_ID, df_setups)

        best = df_setups.iloc[0]

        ok, reason = check_survival_conditions(status, best['pred_ev'], best['volume_spike'])
        if not ok:
            print(f"Trade blocked: {reason}")
        else:
            # Dynamic stop & position sizing
            stop_mult = get_stop_multiplier(status['equity'])
            stop_distance_price = best['atr'] * stop_mult if best['atr'] > 0 else best['price'] * 0.005
            risk_amount = best['risk_amount']
            position_size = risk_amount / stop_distance_price
            min_notional = 1.0
            if position_size * best['price'] < min_notional:
                print(f"Trade skipped: notional too small (${position_size * best['price']:.2f})")
            else:
                stop_price = best['price'] - stop_distance_price if best['direction'] == 'LONG' else best['price'] + stop_distance_price
                # Build Telegram message
                msg = (f"🤖 COZYAI GOD TIER SIGNAL\n"
                       f"Symbol: {best['symbol']}\n"
                       f"Direction: {best['direction']}\n"
                       f"Predicted EV: {best['pred_ev']:.2f}R\n"
                       f"Confidence: {best['confidence']*100:.1f}%\n"
                       f"Risk: {best['risk_pct']*100:.1f}% (${best['risk_amount']:.2f})\n"
                       f"Stop: ${stop_price:.4f}\n"
                       f"Entry: ${best['price']:.4f}\n"
                       f"Zone: {best['fvg_low']:.4f}–{best['fvg_high']:.4f}\n"
                       f"RSI: {best['rsi']:.1f} | MACD: {best['macd']:.2f}\n"
                       f"MC mean: {best['monte_carlo_mean']:.2f} std: {best['monte_carlo_std']:.2f}")
                send_telegram(msg)
                print(f"Found {len(setups)} setups, alerted top.")

                if AUTO_TRADE:
                    order = place_order(best['symbol'], best['direction'], best['price'], position_size)
                    if order:
                        print(f"Auto-trade executed: {order}")
    else:
        print("No setups found.")

if __name__ == "__main__":
    main()
