# src/technicals.py
import pandas as pd
import numpy as np
from typing import Tuple

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()

def macd_components(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series,pd.Series,pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('time').reset_index(drop=True)
    df['EMA200'] = ema(df['close'], 200)
    df['SMA50'] = sma(df['close'], 50)
    macd, macd_signal, macd_hist = macd_components(df['close'])
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    df['RSI'] = rsi(df['close'])
    df['ATR'] = atr(df)
    return df
