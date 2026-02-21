import pandas as pd
import numpy as np
import os
import yfinance as yf
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_FOLDER = "data"
MODEL_FILE = "ml_model.pkl"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# =====================================================
# DATA DOWNLOADER
# =====================================================

def download_data(pair, interval="1h", period="180d"):
    symbol_map = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "XAUUSD": "GC=F"
    }

    symbol = symbol_map.get(pair.upper())
    if not symbol:
        raise Exception("Pair not supported")

    df = yf.download(symbol, interval=interval, period=period)
    df = df.rename(columns=str.lower)
    df.reset_index(inplace=True)

    filename = f"{DATA_FOLDER}/{pair}_{interval}.csv"
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
# SIGNAL GENERATION
# =====================================================

def generate_signal(df):
    latest = df.iloc[-1]

    buy = False
    sell = False

    if latest['ema50'] > latest['ema200']:
        buy = True
    if latest['ema50'] < latest['ema200']:
        sell = True

    if latest['rsi'] < 30:
        buy = True
    if latest['rsi'] > 70:
        sell = True

    if latest['macd'] > latest['macd_signal']:
        buy = True
    else:
        sell = True

    if buy and not sell:
        return 1
    elif sell and not buy:
        return -1
    else:
        return 0

# =====================================================
# BACKTEST ENGINE
# =====================================================

def backtest(pair):
    filename = f"{DATA_FOLDER}/{pair}_1h.csv"
    df = pd.read_csv(filename)
    df = add_indicators(df)

    balance = 10000
    wins = 0
    losses = 0

    for i in range(50, len(df)-1):
        subset = df.iloc[:i]
        signal = generate_signal(subset)

        entry = df.iloc[i]['close']
        next_close = df.iloc[i+1]['close']

        if signal == 1:
            if next_close > entry:
                wins += 1
                balance *= 1.01
            else:
                losses += 1
                balance *= 0.99

        if signal == -1:
            if next_close < entry:
                wins += 1
                balance *= 1.01
            else:
                losses += 1
                balance *= 0.99

    total = wins + losses
    winrate = (wins / total * 100) if total > 0 else 0

    return {
        "final_balance": round(balance,2),
        "wins": wins,
        "losses": losses,
        "winrate": round(winrate,2)
    }

# =====================================================
# MACHINE LEARNING
# =====================================================

def train_ml(pair):
    filename = f"{DATA_FOLDER}/{pair}_1h.csv"
    df = pd.read_csv(filename)
    df = add_indicators(df)

    df['future'] = df['close'].shift(-1)
    df['target'] = (df['future'] > df['close']).astype(int)

    features = df[['ema50','ema200','rsi','macd']]
    target = df['target']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    pickle.dump(model, open(MODEL_FILE,"wb"))

    return round(acc*100,2)

def ml_predict(pair):
    model = pickle.load(open(MODEL_FILE,"rb"))

    filename = f"{DATA_FOLDER}/{pair}_1h.csv"
    df = pd.read_csv(filename)
    df = add_indicators(df)

    latest = df[['ema50','ema200','rsi','macd']].iloc[-1:]
    prob = model.predict_proba(latest)[0][1]

    return round(prob*100,2)
