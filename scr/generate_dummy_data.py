import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import SUPPORTED_PAIRS

def generate_dummy_csv(symbol: str, n=1000):
    date_rng = pd.date_range(end=datetime.today(), periods=n, freq="H")
    df = pd.DataFrame(date_rng, columns=["Date"])
    df["Open"] = np.random.uniform(100, 150, size=n)
    df["High"] = df["Open"] + np.random.uniform(0, 1, size=n)
    df["Low"] = df["Open"] - np.random.uniform(0, 1, size=n)
    df["Close"] = df["Open"] + np.random.uniform(-0.5, 0.5, size=n)
    df.to_csv(f"data/raw/{symbol}.csv", index=False)
    print(f"[OK] Dummy data created: {symbol}.csv")

if __name__ == "__main__":
    for pair in SUPPORTED_PAIRS:
        symbol = pair.replace("/", "")
        generate_dummy_csv(symbol)

