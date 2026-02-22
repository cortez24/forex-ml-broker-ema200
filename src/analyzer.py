import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import warnings

# Abaikan peringatan dari sklearn.utils.parallel yang muncul akibat penggunaan internal
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

DATA_FOLDER = "data"
MODEL_FOLDER = "models"
CONF_THRESHOLD = 60
TIMEFRAME = "4h"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# =========================
# FILE HANDLER
# =========================

def data_file(pair):
    return os.path.join(DATA_FOLDER, f"{pair}_4h.csv")

def model_file(pair):
    return os.path.join(MODEL_FOLDER, f"{pair}_rf.pkl")

def model_meta_file(pair):
    return os.path.join(MODEL_FOLDER, f"{pair}_meta.pkl")


# =========================
# DOWNLOAD 2 YEARS DATA
# =========================

def download_data(pair):
    yahoo_symbol = pair if pair.endswith("=X") else pair + "=X"
    end = datetime.now()
    start = end - timedelta(days=730)

    df = yf.download(
        yahoo_symbol,
        start=start,
        end=end,
        interval="4h",
        auto_adjust=True
    )

    if df.empty:
        raise ValueError("Pair tidak valid atau tidak ada data")

    # HANDLE MultiIndex (yfinance terbaru)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # convert semua kolom jadi string lowercase (aman untuk tuple)
    df.columns = [str(c).lower() for c in df.columns]

    df.to_csv(data_file(pair), index=False)
    return True


# =========================
# INDICATORS
# =========================

def compute_indicators(df):
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
# RULE SCORE
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
# TRAIN MODEL
# =========================

def train_model(pair):
    df = pd.read_csv(data_file(pair))
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

    # Set n_jobs=1 untuk menghindari paralelisasi internal yang memicu warning
    model = RandomForestClassifier(n_estimators=200, n_jobs=1)
    model.fit(X_train, y_train)

    joblib.dump(model, model_file(pair))
    joblib.dump(datetime.now(), model_meta_file(pair))


def ensure_model(pair):
    if not os.path.exists(model_file(pair)):
        train_model(pair)
        return

    last_train = joblib.load(model_meta_file(pair))
    if (datetime.now() - last_train).days > 30:
        train_model(pair)


# =========================
# GENERATE SIGNAL
# =========================

def generate_signal(pair):
    ensure_model(pair)

    df = pd.read_csv(data_file(pair))
    df = compute_indicators(df)

    row = df.iloc[-1]

    if row["adx"] < 18:
        return {"signal": "NO TRADE", "reason": "ADX < 18"}

    trend = 1 if row["ema50"] > row["ema200"] else -1

    model = joblib.load(model_file(pair))

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
# PERFORMANCE REPORT
# =========================

def performance_report(pair):
    ensure_model(pair)

    df = pd.read_csv(data_file(pair))
    df = compute_indicators(df)

    model = joblib.load(model_file(pair))

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
