import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings
from config import MODEL_FOLDER, CONF_THRESHOLD
from data_loader import load_historical_data, get_latest_data_with_realtime

# Abaikan peringatan sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

# =========================
# FILE HANDLER (dengan timeframe)
# =========================

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


# =========================
# RULE SCORE (sama)
# =========================

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
# TRAIN MODEL (dengan timeframe)
# =========================

def train_model(pair, timeframe):
    # Gunakan data historis murni untuk training (tanpa real-time)
    df = load_historical_data(pair, timeframe)
    df = compute_indicators(df)

    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)

    features = [
        "ema_distance", "adx", "rsi",
        "momentum", "rsi_slope",
        "price_vs_ema", "atr"
    ]

    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    # n_jobs=1 untuk menghindari paralelisasi bermasalah
    model = RandomForestClassifier(n_estimators=200, n_jobs=1)
    model.fit(X_train, y_train)

    joblib.dump(model, model_file(pair, timeframe))
    joblib.dump(datetime.now(), model_meta_file(pair, timeframe))

    return model


def ensure_model(pair, timeframe):
    if not os.path.exists(model_file(pair, timeframe)):
        return train_model(pair, timeframe)

    last_train = joblib.load(model_meta_file(pair, timeframe))
    if (datetime.now() - last_train).days > 30:
        return train_model(pair, timeframe)

    return joblib.load(model_file(pair, timeframe))


# =========================
# GENERATE SIGNAL (dengan timeframe dan real-time)
# =========================

def generate_signal(pair, timeframe):
    # Pastikan model ada
    model = ensure_model(pair, timeframe)

    # Ambil data terbaru dengan real-time
    df = get_latest_data_with_realtime(pair, timeframe)
    df = compute_indicators(df)

    if df.empty:
        return {"signal": "NO TRADE", "reason": "Data kosong"}

    row = df.iloc[-1]

    if row["adx"] < 18:
        return {"signal": "NO TRADE", "reason": "ADX < 18"}

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
        "tp2": round(tp2, 5)
    }


# =========================
# PERFORMANCE REPORT (dengan timeframe)
# =========================

def performance_report(pair, timeframe):
    ensure_model(pair, timeframe)

    # Gunakan data historis saja untuk backtest (tidak termasuk real-time)
    df = load_historical_data(pair, timeframe)
    df = compute_indicators(df)

    model = joblib.load(model_file(pair, timeframe))

    trades = []
    equity_curve = []
    equity = 0

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
    # ... (semua fungsi yang sudah ada) ...

def train_all_models(force=False):
    """
    Melatih model untuk semua pair dan timeframe.
    Jika force=False, hanya melatih model yang sudah lebih dari TRAINING_INTERVAL_DAYS.
    """
    from config import SUPPORTED_PAIRS, TIMEFRAMES, TRAINING_INTERVAL_DAYS
    import os
    from datetime import datetime

    trained_count = 0
    for pair in SUPPORTED_PAIRS:
        for tf in TIMEFRAMES:
            meta_file = model_meta_file(pair, tf)
            need_train = False
            if not os.path.exists(model_file(pair, tf)):
                need_train = True
            elif force:
                need_train = True
            else:
                # Cek umur model
                try:
                    last_train = joblib.load(meta_file)
                    if (datetime.now() - last_train).days > TRAINING_INTERVAL_DAYS:
                        need_train = True
                except:
                    need_train = True

            if need_train:
                print(f"Training model untuk {pair} {tf}...")
                train_model(pair, tf)
                trained_count += 1
    return trained_count
