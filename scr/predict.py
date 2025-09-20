import os
import joblib
import pandas as pd
from feature_engineering import prepare_features


def load_model(pair: str, model_dir="models"):
    """
    Load model terlatih untuk pasangan forex tertentu.
    """
    model_path = os.path.join(model_dir, f"{pair}_xgb.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {pair} tidak ditemukan di {model_path}")
    return joblib.load(model_path)


def predict_next_move(pair: str, data_dir="data/processed", model_dir="models"):
    """
    Prediksi arah harga berikutnya untuk pasangan forex tertentu.
    Return: dict dengan hasil prediksi + probabilitas
    """
    # Load data harga
    file_path = os.path.join(data_dir, f"{pair}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data {pair} tidak ditemukan di {file_path}")

    df_price = pd.read_csv(file_path)
    df_price["time"] = pd.to_datetime(df_price["time"])

    # Load data news
    news_path = os.path.join(data_dir, "news.csv")
    df_news = pd.read_csv(news_path) if os.path.exists(news_path) else None

    # Feature engineering
    df = prepare_features(df_price, df_news)

    # Load model
    model = load_model(pair, model_dir)

    # Gunakan hanya data terakhir
    latest = df.drop(columns=["time"]).iloc[-1:]
    prob = model.predict_proba(latest)[0]
    pred = model.predict(latest)[0]

    result = {
        "pair": pair,
        "prediction": "UP" if pred == 1 else "DOWN",
        "probability_up": float(prob[1]),
        "probability_down": float(prob[0]),
        "last_price": float(df_price["close"].iloc[-1]),
        "timestamp": str(df_price["time"].iloc[-1])
    }
    return result


if __name__ == "__main__":
    pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
    for pair in pairs:
        try:
            res = predict_next_move(pair)
            print(res)
        except Exception as e:
            print(f"[ERROR] {pair}: {e}")

