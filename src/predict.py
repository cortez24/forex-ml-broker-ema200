# src/predict.py
import os
import joblib
import pandas as pd
from feature_engineering import load_price, load_news, add_technicals, prepare_and_save
from config import MODELS_DIR, PROCESSED_DIR
from tpsl import map_prob_to_tpsl

def load_model_for(symbol: str):
    path = os.path.join(MODELS_DIR, f"{symbol}_xgb.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return joblib.load(path)

def predict_next(symbol: str):
    """
    symbol e.g. 'EURUSD'
    returns dict with prob_up, prob_down, pred_label, recommended tp/sl (abs levels)
    """
    model_path = os.path.join(MODELS_DIR, f"{symbol}_xgb.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train first.")
    model = joblib.load(model_path)
    # ensure features exist; if not, create
    features_path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
    if not os.path.exists(features_path):
        prepare_and_save(symbol)
    df = pd.read_csv(features_path)
    X = df.drop(columns=['time','target'], errors='ignore')
    last_feat = X.iloc[-1:]
    probs = model.predict_proba(last_feat)[0]
    prob_up = float(probs[1])
    prob_down = float(probs[0])
    pred = 1 if prob_up > prob_down else 0
    last_price = df['close'].iloc[-1]
    tl = map_prob_to_tpsl(prob_up if pred==1 else prob_down)
    if tl is not None:
        if tl['direction'] == 'long':
            tp_price = last_price*(1+tl['tp_pct'])
            sl_price = last_price*(1-tl['sl_pct'])
        else:
            tp_price = last_price*(1-tl['tp_pct'])
            sl_price = last_price*(1+tl['sl_pct'])
    else:
        tp_price = sl_price = None
    return {
        'symbol': symbol,
        'prediction': 'UP' if pred==1 else 'DOWN',
        'prob_up': prob_up,
        'prob_down': prob_down,
        'last_price': last_price,
        'tp_price': tp_price,
        'sl_price': sl_price
    }

if __name__ == "__main__":
    print(predict_next("EURUSD"))
