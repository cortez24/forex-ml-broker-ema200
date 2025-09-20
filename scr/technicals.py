import pandas as pd
import ta

def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["ema_200"] = ta.trend.EMAIndicator(df["Close"], window=200).ema_indicator()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    return df
