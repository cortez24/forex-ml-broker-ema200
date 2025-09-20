import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_engineering import prepare_features


# ==============================
# PIPELINE
# ==============================

def train_model(pair: str, data_dir="data/processed", model_dir="models"):
    """
    Train model untuk 1 pasangan forex (misalnya EURUSD).
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

    # Target label: arah harga (1 naik, 0 turun)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()

    X = df.drop(columns=["time", "target"])
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"=== Report {pair} ===")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, f"{pair}_xgb.json")
    joblib.dump(model, model_file)
    print(f"[OK] Model {pair} disimpan di {model_file}")


def train_all_pairs(pairs=None):
    """
    Train semua pasangan forex yang tersedia.
    """
    if pairs is None:
        pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]

    for pair in pairs:
        train_model(pair)


if __name__ == "__main__":
    train_all_pairs()

