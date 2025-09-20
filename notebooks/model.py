import xgboost as xgb
from sklearn.metrics import classification_report
import pandas as pd
from features import add_macd, add_ema200, add_labels

def train_xgb(df):
    df = add_macd(df)
    df = add_ema200(df)
    df = add_labels(df).dropna()

    X = df[["MACD", "Signal", "EMA200"]]
    y = df["Target"]

    dtrain = xgb.DMatrix(X, label=y)
    params = {"objective": "binary:logistic", "eval_metric": "logloss"}
    model = xgb.train(params, dtrain, num_boost_round=100)

    preds = (model.predict(dtrain) > 0.5).astype(int)
    print(classification_report(y, preds))
    return model

if __name__ == "__main__":
    df = pd.read_csv("../data/forex/EURUSD_4H.csv")
    model = train_xgb(df)
    model.save_model("../data/model_eurusd_4h.json")
    print("✅ Model with EMA200 saved!")

