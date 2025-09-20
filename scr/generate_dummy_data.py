# src/generate_dummy_data.py
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import SUPPORTED_PAIRS, RAW_DIR

os.makedirs(RAW_DIR, exist_ok=True)

def generate(pair: str, bars=500, freq_hours=4):
    symbol = pair.replace('/','')
    end = datetime.utcnow()
    periods = bars
    dates = pd.date_range(end=end, periods=periods, freq=f"{freq_hours}H")
    # start price by pair heuristics (JPY pairs ~ 140, EURUSD~1.1, USDIDR~15000)
    if 'JPY' in pair:
        base = 140.0
    elif 'IDR' in pair:
        base = 15000.0
    elif 'EUR' in pair and 'USD' in pair:
        base = 1.1
    else:
        base = 1.0
    prices = [base]
    for _ in range(1, len(dates)):
        change = np.random.normal(0, 0.002)
        prices.append(prices[-1]*(1+change))
    df = pd.DataFrame({
        'time': dates,
        'open': [p*(1+np.random.uniform(-0.0005,0.0005)) for p in prices],
        'high': [p*(1+np.random.uniform(0.0001,0.003)) for p in prices],
        'low': [p*(1-np.random.uniform(0.0001,0.003)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100,1000,len(dates))
    })
    out = os.path.join(RAW_DIR, f"{symbol}.csv")
    df.to_csv(out, index=False)
    print(f"[OK] dummy {symbol} -> {out}")

if __name__ == "__main__":
    for p in SUPPORTED_PAIRS:
        generate(p, bars=1000)
