import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODEL_FILE = os.path.join(BASE_DIR, "ml_model.pkl")

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# =====================================================
# LOAD FILE KHUSUS FORMAT ANDA (TAB, TANPA HEADER)
# =====================================================

def load_price_file(filename):

    df = pd.read_csv(filename, sep="\t", header=None)

    df.columns = ['datetime','open','high','low','close','volume']

    df.columns = df.columns.str.lower()

    numeric_cols = ['open','high','low','close','volume']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    return df

# =====================================================
# INDICATORS
# =====================================================

def add_indicators(df):

    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['close'].ewm(span=12).mean()
    exp2 = df['close'].ewm(span=26).mean()

    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    df.dropna(inplace=True)

    return df

# =====================================================
# MACHINE LEARNING
# =====================================================

def train_ml(pair):

    pair = pair.upper()
    filename = os.path.join(DATA_FOLDER, f"{pair}_1h.csv")

    if not os.path.exists(filename):
        raise Exception(f"File {filename} tidak ditemukan")

    df = load_price_file(filename)
    df = add_indicators(df)

    df['future'] = df['close'].shift(-1)
    df['target'] = (df['future'] > df['close']).astype(int)

    df.dropna(inplace=True)

    features = df[['ema50','ema200','rsi','macd']]
    target = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    pickle.dump(model, open(MODEL_FILE,"wb"))

    return round(acc*100,2)

def ml_predict(pair):

    if not os.path.exists(MODEL_FILE):
        raise Exception("Model belum di-train")

    pair = pair.upper()
    filename = os.path.join(DATA_FOLDER, f"{pair}_1h.csv")

    df = load_price_file(filename)
    df = add_indicators(df)

    model = pickle.load(open(MODEL_FILE,"rb"))

    latest = df[['ema50','ema200','rsi','macd']].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]

    return round(prob*100,2)
