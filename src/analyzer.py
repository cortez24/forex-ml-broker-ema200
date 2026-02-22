import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
from config import MODEL_FOLDER, CONF_THRESHOLD, SIGNAL_TIMEFRAME, PAIRS, TIMEFRAMES
from data_loader import load_historical_data, get_realtime_price

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

# Pastikan folder models ada
os.makedirs(MODEL_FOLDER, exist_ok=True)

def model_file(pair, timeframe):
    return os.path.join(MODEL_FOLDER, f"{pair}_{timeframe}_rf.pkl")

def model_meta_file(pair, timeframe):
    return os.path.join(MODEL_FOLDER, f"{pair}_{timeframe}_meta.pkl")

# =========================
# INDICATORS (sama seperti sebelumnya)
# =========================
def compute_indicators(df):
    df = df.copy()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["ema_distance"] = df["ema50"] - df["ema200"]
    df["rsi"] = compute_rsi(df["close"], 14)
    df["momentum"] = df["close"].diff(5)
    df["rsi_slope"] = df["rsi"].diff()
    df["price_vs_ema"] = df["close"] - df["ema50"]
    df["atr"] = compute_atr(df, 14)
    df["adx"] = compute_adx(df, 14)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_atr(df, period):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    return tr.rolling(period).mean()

def compute_adx(df, period):
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff().abs()
    tr = compute_atr(df, period)
    plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr)
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    return dx.rolling(period).mean()

def rule_score(row):
    score = 0
    if row["ema50"] > row["ema200"]:
        score += 30
    else:
        score -= 30
    if 40 < row["rsi"] < 60:
        score += 20
    if row["adx"] > 20:
        score += 20
    return max(0, min(100, score))

# =========================
# TRAIN MODEL
# =========================
def train_model(pair, timeframe=SIGNAL_TIMEFRAME):
    df = load_historical_data(pair, timeframe)
    df = compute_indicators(df)

    # Target: apakah harga berikutnya naik (close next > close)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    features = [
        "ema_distance", "adx", "rsi",
        "momentum", "rsi_slope",
        "price_vs_ema", "atr"
    ]

    X = df[features]
    y = df["target"]

    # Bagi data tanpa shuffle (time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=200, n_jobs=1)
    model.fit(X_train, y_train)

    joblib.dump(model, model_file(pair, timeframe))
    joblib.dump(datetime.now(), model_meta_file(pair, timeframe))

def ensure_model(pair, timeframe=SIGNAL_TIMEFRAME):
    model_path = model_file(pair, timeframe)
    meta_path = model_meta_file(pair, timeframe)
    
    if not os.path.exists(model_path):
        train_model(pair, timeframe)
        return

    # Periksa umur model
    last_train = joblib.load(meta_path)
    if (datetime.now() - last_train).days > 30:
        train_model(pair, timeframe)

# =========================
# GENERATE SIGNAL (menggabungkan data historis + real-time)
# =========================
def generate_signal(pair, timeframe=SIGNAL_TIMEFRAME):
    # Pastikan model tersedia
    ensure_model(pair, timeframe)

    # Ambil data historis terbaru (untuk indikator)
    df_hist = load_historical_data(pair, timeframe)
    df_hist = compute_indicators(df_hist)

    # Ambil harga real-time
    rt = get_realtime_price(pair)
    if rt is None:
        # Jika gagal, gunakan candle terakhir dari historis (kurang akurat)
        row = df_hist.iloc[-1]
    else:
        # Buat row gabungan: indikator dari historis terakhir, harga dari real-time
        last_hist = df_hist.iloc[-1]
        # Kita asumsikan indikator masih relevan (untuk sementara)
        # Sebenarnya perlu hitung ulang indikator dengan data real-time jika memungkinkan,
        # tapi untuk sederhana, kita gunakan nilai indikator dari candle historis terakhir.
        # Alternatif: bisa menghitung indikator berdasarkan data real-time jika kita punya cukup data.
        # Di sini kita hanya menggunakan harga real-time untuk entry.
        row = last_hist.copy()
        row['close'] = rt['close']
        row['high'] = rt['high']
        row['low'] = rt['low']
        row['open'] = rt['open']
        row['volume'] = rt['volume']
        # Catatan: indikator seperti EMA, RSI, ATR, ADX tidak dihitung ulang.
        # Untuk akurasi lebih, sebaiknya kumpulkan data real-time dan hitung ulang indikator.

    if row["adx"] < 18:
        return {"signal": "NO TRADE", "reason": "ADX < 18"}

    trend = 1 if row["ema50"] > row["ema200"] else -1

    model = joblib.load(model_file(pair, timeframe))

    features = pd.DataFrame([[
        row["ema_distance"], row["adx"], row["rsi"],
        row["momentum"], row["rsi_slope"],
        row["price_vs_ema"], row["atr"]
    ]], columns=[
        "ema_distance", "adx", "rsi",
        "momentum", "rsi_slope",
        "price_vs_ema", "atr"
    ])

    ml_prob = model.predict_proba(features)[0][1]
    rule = rule_score(row)

    final_conf = (rule * 0.6) + (ml_prob * 100 * 0.4)

    if final_conf < CONF_THRESHOLD:
        return {"signal": "NO TRADE", "confidence": round(final_conf, 2)}

    entry = row["close"]
    sl = entry - (1.2 * row["atr"] * trend)
    tp1 = entry + (1.5 * row["atr"] * trend)
    tp2 = entry + (2.5 * row["atr"] * trend * ml_prob)

    direction = "BUY" if trend == 1 else "SELL"

    return {
        "signal": direction,
        "confidence": round(final_conf, 2),
        "entry": round(entry, 5),
        "sl": round(sl, 5),
        "tp1": round(tp1, 5),
        "tp2": round(tp2, 5),
        "real_time_used": rt is not None
    }

# =========================
# PERFORMANCE REPORT (tetap menggunakan data historis saja)
# =========================
def performance_report(pair, timeframe=SIGNAL_TIMEFRAME):
    ensure_model(pair, timeframe)

    df = load_historical_data(pair, timeframe)
    df = compute_indicators(df)

    model = joblib.load(model_file(pair, timeframe))

    trades = []
    equity_curve = []
    equity = 0

    # Loop untuk backtest sederhana (gunakan data dari indeks 200 ke atas)
    for i in range(200, len(df)-1):
        row = df.iloc[i]

        if row["adx"] < 18:
            continue

        trend = 1 if row["ema50"] > row["ema200"] else -1

        features = pd.DataFrame([[
            row["ema_distance"], row["adx"], row["rsi"],
            row["momentum"], row["rsi_slope"],
            row["price_vs_ema"], row["atr"]
        ]], columns=[
            "ema_distance", "adx", "rsi",
            "momentum", "rsi_slope",
            "price_vs_ema", "atr"
        ])

        ml_prob = model.predict_proba(features)[0][1]
        rule = rule_score(row)
        final_conf = (rule * 0.6) + (ml_prob * 100 * 0.4)

        if final_conf < CONF_THRESHOLD:
            continue

        entry = row["close"]
        next_close = df.iloc[i+1]["close"]

        result = (next_close - entry) * trend

        trades.append(result)
        equity += result
        equity_curve.append(equity)

    if len(trades) == 0:
        return {"error": "No trades"}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    winrate = len(wins)/len(trades)
    sharpe = np.mean(trades)/np.std(trades) if np.std(trades) != 0 else 0

    return {
        "metrics": {
            "total_trades": len(trades),
            "winrate": round(winrate*100, 2),
            "sharpe": round(sharpe, 2)
        },
        "equity_curve": equity_curve
    }

# Untuk kemudahan, bisa juga ditambahkan fungsi yang mengembalikan data historis terakhir
def get_latest_candle(pair, timeframe):
    df = load_historical_data(pair, timeframe)
    if df.empty:
        return None
    last = df.iloc[-1].to_dict()
    last['datetime'] = last['datetime'].isoformat()
    return last
