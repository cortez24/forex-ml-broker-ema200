import pandas as pd
import yfinance as yf

def fetch_price_pair(pair="EUR/USD", interval="240min"):
    # Mapping ke simbol Yahoo Finance
    symbol_map = {
        "EUR/USD": "EURUSD=X",
        "USD/IDR": "USDIDR=X",
        "AUD/USD": "AUDUSD=X"
    }
    if pair not in symbol_map:
        raise ValueError(f"Pair {pair} not supported")

    ticker = symbol_map[pair]
    # Ambil data 6 bulan terakhir, timeframe 4 jam (240m)
    df = yf.download(ticker, period="6mo", interval="240m")
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
    df.dropna(inplace=True)
    return df

