# src/technicals.py
"""
Module indikator teknikal forex.
Berisi fungsi EMA, SMA, MACD, dan wrapper untuk preprocessing.
"""

import pandas as pd

# === Exponential Moving Average (EMA) ===
def ema(df, column="close", period=200):
    """
    Hitung EMA.
    """
    return df[column].ewm(span=period, adjust=False).mean()

# === Simple Moving Average (SMA) ===
def sma(df, column="close", period=50):
    """
    Hitung SMA.
    """
    return df[column].rolling(window=period).mean()

# === Moving Average Convergence Divergence (MACD) ===
def macd(df, column="close", fast=12, slow=26, signal=9):
    """
    Hitung indikator MACD.
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd_line": macd_line,
        "signal_line": signal_line,
        "macd_histogram": histogram
    })

# === Wrapper untuk menambahkan indikator teknikal ===
def add_indicators(df):
    """
    Tambahkan EMA200, SMA50, MACD ke DataFrame harga forex.
    """
    df = df.copy()

    # EMA200
    df["ema200"] = ema(df, period=200)

    # SMA50
    df["sma50"] = sma(df, period=50)

    # MACD
    macd_df = macd(df)
    df = pd.concat([df, macd_df], axis=1)

    return df

