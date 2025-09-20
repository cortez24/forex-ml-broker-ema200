# src/feature_engineering.py
import os
import pandas as pd
import numpy as np
from typing import Optional
from technicals import add_technicals
from config import PROCESSED_DIR, RAW_DIR

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_price(symbol: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{symbol}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=['time'])
    # Normalize column names lower
    df.columns = [c.lower() for c in df.columns]
    # Ensure 'time','open','high','low','close','volume' exist
    return df.rename(columns={'time':'time','open':'open','high':'high','low':'low','close':'close','volume':'volume'})

def load_news() -> Optional[pd.DataFrame]:
    npath = os.path.join(RAW_DIR, "news.csv")
    if not os.path.exists(npath):
        return None
    news = pd.read_csv(npath, parse_dates=['time'])
    return news

def aggregate_news_features(price_df: pd.DataFrame, news_df: pd.DataFrame, window='1D') -> pd.DataFrame:
    # simple: create count of news in prior 24h and mean sentiment if sentiment exists
    news_df = news_df.copy()
    news_df['time'] = pd.to_datetime(news_df['time'])
    price_df = price_df.copy()
    price_df['time'] = pd.to_datetime(price_df['time'])
    news_df = news_df.sort_values('time')
    price_df = price_df.sort_values('time')
    # For speed, use merge_asof to attach last news timestamp and then count within window
    merged = pd.merge_asof(price_df, news_df[['time','title']], left_on='time', right_on='time', direction='backward')
    # compute simple counts in the prior window
    # (inefficient for huge data, ok for prototyping)
    counts = []
    for t in price_df['time']:
        start = t - pd.Timedelta(window)
        counts.append(((news_df['time'] >= start) & (news_df['time'] <= t)).sum())
    price_df['news_count_1d'] = counts
    return price_df

def prepare_and_save(symbol: str):
    # symbol: 'EURUSD'
    price = load_price(symbol)
    news = load_news()
    price = add_technicals(price)
    if news is not None:
        price = aggregate_news_features(price, news, window='1D')
    # create target
    price['target'] = (price['close'].shift(-1) > price['close']).astype(int)
    # drop rows with NaN due to indicators
    price = price.dropna().reset_index(drop=True)
    out_path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
    price.to_csv(out_path, index=False)
    print(f"[OK] saved features for {symbol} -> {out_path}")
    return out_path

if __name__ == "__main__":
    # process all supported pairs found in RAW_DIR
    files = os.listdir(RAW_DIR)
    symbols = [f.split('.')[0] for f in files if f.endswith('.csv') and not f.startswith('news')]
    for s in symbols:
        try:
            prepare_and_save(s)
        except Exception as e:
            print(f"[ERROR] preparing {s}: {e}")
