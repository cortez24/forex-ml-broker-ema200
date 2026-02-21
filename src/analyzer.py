import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_FOLDER = "data"
MODEL_FOLDER = "models"
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

CONF_THRESHOLD = 62
RETRAIN_DAYS = 30


# =========================
# UTIL
# =========================

def normalize_pair(pair):
    return pair.upper().replace("_4H","").replace("=X","")


def yahoo_symbol(pair):
    pair = normalize_pair(pair)
    if pair == "XAUUSD":
        return "GC=F"
    return pair + "=X"


def data_file(pair):
    return os.path.join(DATA_FOLDER, f"{normalize_pair(pair)}_4h.csv")


def model_file(pair):
    return os.path.join(MODEL_FOLDER, f"{normalize_pair(pair)}.pkl")


# =========================
# DOWNLOAD 2 YEARS
# =========================

def download_data(pair):
    symbol = yahoo_symbol(pair)
    df = yf.download(symbol, interval="4h", period="2y")

    if df.empty:
        raise Exception("Download gagal")

    df = df.rename(columns=str.lower)
    df.reset_index(inplace=True)
    df.to_csv(data_file(pair), index=False)
    return "OK"


# =========================
# INDICATORS
# =========================

def compute_indicators(df):

    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100/(1+rs))

    high_low = df["high"] - df["low"]
    high_close = abs(df["high"] - df["close"].shift())
    low_close = abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = abs(100 * (minus_dm.rolling(14).sum() / tr14))

    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    df["adx"] = dx.rolling(14).mean()

    df["ema_distance"] = abs(df["ema50"] - df["ema200"])
    df["momentum"] = df["close"] - df["close"].shift(3)
    df["rsi_slope"] = df["rsi"] - df["rsi"].shift(3)
    df["price_vs_ema"] = df["close"] - df["ema50"]

    df.dropna(inplace=True)
    return df


# =========================
# TRAIN MODEL
# =========================

def train_model(pair):

    df = pd.read_csv(data_file(pair))
    df = compute_indicators(df)

    df["target"] = np.where(df["close"].shift(-1) > df["close"],1,0)
    df.dropna(inplace=True)

    features = df[[
        "ema_distance","adx","rsi",
        "momentum","rsi_slope",
        "price_vs_ema","atr"
    ]]

    target = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, model_file(pair))
    return "trained"


# =========================
# AUTO RETRAIN CHECK
# =========================

def ensure_model(pair):

    file = model_file(pair)

    if not os.path.exists(file):
        train_model(pair)
        return

    created = datetime.fromtimestamp(os.path.getmtime(file))
    if datetime.now() - created > timedelta(days=RETRAIN_DAYS):
        train_model(pair)


# =========================
# RULE SCORE
# =========================

def rule_score(row):

    score = 0

    if row["ema_distance"] > row["atr"]*0.5:
        score += 30

    if row["adx"] > 25:
        score += 30
    elif row["adx"] > 18:
        score += 20

    if 40 < row["rsi"] < 60:
        score += 20

    return min(score,100)


# =========================
# SIGNAL GENERATOR
# =========================

def generate_signal(pair):

    if not os.path.exists(data_file(pair)):
        raise Exception("Data belum didownload")

    ensure_model(pair)

    df = pd.read_csv(data_file(pair))
    df = compute_indicators(df)

    last = df.iloc[-1]

    if last["adx"] < 18:
        return {"pair":pair,"signal":"NO TRADE (Low ADX)"}

    trend = None
    if last["ema50"] > last["ema200"]:
        trend = "BUY"
    elif last["ema50"] < last["ema200"]:
        trend = "SELL"
    else:
        return {"pair":pair,"signal":"NO TREND"}

    model = joblib.load(model_file(pair))

    feature_row = pd.DataFrame([[
        last["ema_distance"],last["adx"],last["rsi"],
        last["momentum"],last["rsi_slope"],
        last["price_vs_ema"],last["atr"]
    ]],columns=[
        "ema_distance","adx","rsi",
        "momentum","rsi_slope",
        "price_vs_ema","atr"
    ])

    ml_prob = model.predict_proba(feature_row)[0][1]

    rule = rule_score(last)
    final_conf = (rule*0.6) + (ml_prob*100*0.4)

    if final_conf < CONF_THRESHOLD:
        return {"pair":pair,"signal":"NO TRADE (Low Confidence)"}

    atr = last["atr"]
    ema = last["ema50"]

    if trend == "BUY":
        entry_low = ema - 0.2*atr
        entry_high = ema + 0.2*atr
        sl = entry_low - 1.2*atr
        tp1 = entry_high + 1.0*atr

        if ml_prob > 0.75:
            tp2 = entry_high + 2.4*atr
        elif ml_prob > 0.65:
            tp2 = entry_high + 2.0*atr
        else:
            tp2 = entry_high + 1.6*atr

    else:
        entry_low = ema - 0.2*atr
        entry_high = ema + 0.2*atr
        sl = entry_high + 1.2*atr
        tp1 = entry_low - 1.0*atr

        if ml_prob > 0.75:
            tp2 = entry_low - 2.4*atr
        elif ml_prob > 0.65:
            tp2 = entry_low - 2.0*atr
        else:
            tp2 = entry_low - 1.6*atr

    return {
        "pair":pair,
        "bias":trend,
        "entry_zone":[round(entry_low,5),round(entry_high,5)],
        "sl":round(sl,5),
        "tp1":round(tp1,5),
        "tp2":round(tp2,5),
        "confidence":round(final_conf,2),
        "ml_probability":round(ml_prob,3),
        "adx":round(last["adx"],2)
    }


# =========================
# MARKET SCANNER
# =========================

def scan_market():

    pairs = [
        "EURUSD","GBPUSD","USDJPY",
        "AUDUSD","NZDUSD","USDCAD",
        "XAUUSD"
    ]

    results = []

    for pair in pairs:
        try:
            sig = generate_signal(pair)
            if "bias" in sig:
                results.append(sig)
        except:
            continue

    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results
