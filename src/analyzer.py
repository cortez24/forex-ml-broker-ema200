import pandas as pd
import numpy as np
import os
import yfinance as yf
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")
MODEL_FILE = os.path.join(BASE_DIR, "ml_model.pkl")

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# =====================================================
# SMART CSV LOADER (AUTO DETECT FORMAT)
# =====================================================

def load_price_file(filename):

    # auto detect separator
    df = pd.read_csv(filename, sep=None, engine="python")

    # jika cuma 1 kolom → coba whitespace
    if df.shape[1] == 1:
        df = pd.read_csv(filename, delim_whitespace=True)

    # jika masih 1 kolom → file rusak
    if df.shape[1] < 6:
        raise Exception("Format CSV salah. Harus 6 kolom.")

    # jika tidak ada header close → berarti tanpa header
    if "close" not in df.columns:
        df = pd.read_csv(filename, sep=None, engine="python", header=None)

        if df.shape[1] == 1:
            df = pd.read_csv(filename, delim_whitespace=True, header=None)

        df.columns = ["datetime","open","high","low","close","volume"]

    df.columns = df.columns.str.lower()

    numeric_cols = ["open","high","low","close","volume"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    return df

# =====================================================
# DOWNLOAD DATA FROM YAHOO
# =====================================================

def download_data(pair, interval="1h", period="180d"):

    if not pair:
        raise ValueError("Pair kosong")

    pair = pair.upper().replace("_1H","").replace("_1h","")

    symbol_map = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "XAUUSD": "GC=F"
    }

    if pair.endswith("=X") or pair.endswith("=F"):
        symbol = pair
    else:
        symbol = symbol_map.get(pair)

    if not symbol:
        raise ValueError(f"Pair {pair} tidak didukung")

    print("Downloading:", symbol)

    df = yf.download(symbol, interval=interval, period=period)

    if df.empty:
        raise Exception("Download gagal dari Yahoo")

    df = df.rename(columns=str.lower)
    df.reset_index(inplace=True)

    filename = os.path.join(DATA_FOLDER, f"{pair}_{interval}.csv")
    df.to_csv(filename, index=False)

    return filename

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
# BACKTEST
# =====================================================

def backtest(pair):

    pair = pair.upper()
    filename = os.path.join(DATA_FOLDER, f"{pair}_1h.csv")

    if not os.path.exists(filename):
        raise Exception("File data tidak ditemukan")

    df = load_price_file(filename)
    df = add_indicators(df)

    balance = 10000
    wins = 0
    losses = 0

    for i in range(50, len(df)-1):

        entry = df.iloc[i]['close']
        next_close = df.iloc[i+1]['close']

        if df.iloc[i]['ema50'] > df.iloc[i]['ema200']:
            signal = 1
        else:
            signal = -1

        if signal == 1 and next_close > entry:
            wins += 1
            balance *= 1.01
        elif signal == -1 and next_close < entry:
            wins += 1
            balance *= 1.01
        else:
            losses += 1
            balance *= 0.99

    total = wins + losses
    winrate = (wins / total * 100) if total > 0 else 0

    return {
        "balance": round(balance,2),
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate,2)
    }

# =====================================================
# MACHINE LEARNING TRAIN
# =====================================================

def train_ml(pair):

    pair = pair.upper()
    filename = os.path.join(DATA_FOLDER, f"{pair}_1h.csv")

    if not os.path.exists(filename):
        raise Exception("File data tidak ditemukan")

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

    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    pickle.dump(model, open(MODEL_FILE,"wb"))

    return round(acc*100,2)

# =====================================================
# MACHINE LEARNING PREDICT
# =====================================================

def ml_predict(pair):

    if not os.path.exists(MODEL_FILE):
        raise Exception("Model belum di-train")

    pair = pair.upper()
    filename = os.path.join(DATA_FOLDER, f"{pair}_1h.csv")

    if not os.path.exists(filename):
        raise Exception("File data tidak ditemukan")

    df = load_price_file(filename)
    df = add_indicators(df)

    model = pickle.load(open(MODEL_FILE,"rb"))

    latest = df[['ema50','ema200','rsi','macd']].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]

    return round(prob*100,2)
