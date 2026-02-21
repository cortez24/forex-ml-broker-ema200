import os
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DATA_FOLDER = "data"
MODEL_FOLDER = "models"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


# ===============================
# UTIL
# ===============================

def normalize_pair(pair):
    if not pair:
        raise ValueError("Pair kosong")

    pair = pair.upper()
    pair = pair.replace("_1H", "")
    pair = pair.replace("=X", "")
    pair = pair.replace("=F", "")

    return pair


def get_yahoo_symbol(pair):
    pair = normalize_pair(pair)

    # Gold
    if pair == "XAUUSD":
        return "GC=F"

    # Semua forex otomatis
    return pair + "=X"


def get_filename(pair, interval="1h"):
    pair = normalize_pair(pair)
    return os.path.join(DATA_FOLDER, f"{pair}_{interval}.csv")


# ===============================
# DOWNLOAD
# ===============================

def download_data(pair, interval="1h", period="180d"):
    pair = normalize_pair(pair)
    symbol = get_yahoo_symbol(pair)

    print("Downloading:", symbol)

    df = yf.download(symbol, interval=interval, period=period)

    if df.empty:
        raise Exception("Download gagal dari Yahoo Finance")

    df = df.rename(columns=str.lower)
    df.reset_index(inplace=True)

    filename = get_filename(pair, interval)
    df.to_csv(filename, index=False)

    print("Saved to:", filename)
    return filename


# ===============================
# LOAD FILE
# ===============================

def load_price_file(filename):
    if not os.path.exists(filename):
        raise Exception("File tidak ditemukan")

    df = pd.read_csv(filename)

    required_cols = ['open','high','low','close','volume']
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"Kolom {col} tidak ditemukan")

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    return df


# ===============================
# INDICATORS
# ===============================

def add_indicators(df):
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()

    df['rsi'] = compute_rsi(df['close'], 14)

    df.dropna(inplace=True)
    return df


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ===============================
# BACKTEST
# ===============================

def backtest(pair):
    filename = get_filename(pair)

    df = load_price_file(filename)
    df = add_indicators(df)

    df['signal'] = np.where(df['ema50'] > df['ema200'], 1, -1)
    df['returns'] = df['close'].pct_change()
    df['strategy'] = df['signal'].shift(1) * df['returns']

    total_return = df['strategy'].sum()

    return {"total_return": float(total_return)}


# ===============================
# MACHINE LEARNING
# ===============================

def train_ml(pair):
    filename = get_filename(pair)

    df = load_price_file(filename)
    df = add_indicators(df)

    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df.dropna(inplace=True)

    features = df[['ema50','ema200','rsi']]
    target = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    model_file = os.path.join(MODEL_FOLDER, f"{normalize_pair(pair)}.pkl")
    joblib.dump(model, model_file)

    return float(acc)


def ml_predict(pair):
    filename = get_filename(pair)
    model_file = os.path.join(MODEL_FOLDER, f"{normalize_pair(pair)}.pkl")

    if not os.path.exists(model_file):
        raise Exception("Model belum dilatih")

    df = load_price_file(filename)
    df = add_indicators(df)

    latest = df[['ema50','ema200','rsi']].iloc[-1:]

    model = joblib.load(model_file)
    prob = model.predict_proba(latest)[0][1]

    return float(prob)
