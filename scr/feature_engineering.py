import pandas as pd
from config import SUPPORTED_PAIRS
from technicals import add_technicals

def prepare_features(pair: str, input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df = add_technicals(df)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Features generated for {pair}")

if __name__ == "__main__":
    for pair in SUPPORTED_PAIRS:
        symbol = pair.replace("/", "")
        prepare_features(
            pair,
            f"data/raw/{symbol}.csv",
            f"data/processed/{symbol}_features.csv"
        )
