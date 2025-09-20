# src/ml_pipeline.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from config import PROCESSED_DIR, MODELS_DIR
from typing import List

os.makedirs(MODELS_DIR, exist_ok=True)

def load_features(symbol: str) -> pd.DataFrame:
    p = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    return df

def build_pipeline():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1, tree_method='hist'))
    ])
    return pipe

def hyperparam_search(X, y, n_iter=30):
    pipe = build_pipeline()
    param_dist = {
        'clf__max_depth': [3,5,7,9],
        'clf__learning_rate': [0.01, 0.03, 0.05, 0.1],
        'clf__n_estimators': [100,200,300],
        'clf__subsample': [0.6,0.8,1.0],
        'clf__colsample_bytree': [0.6,0.8,1.0]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    rsearch = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
    rsearch.fit(X, y)
    return rsearch

def train_symbol(symbol: str, do_search: bool = False):
    df = load_features(symbol)
    X = df.drop(columns=['time','target'])
    y = df['target']
    # simple split
    split_idx = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if do_search:
        print(f"[{symbol}] Starting hyperparam search ...")
        search = hyperparam_search(X_train, y_train)
        best = search.best_estimator_
        print(f"[{symbol}] best params: {search.best_params_}")
        model = best
    else:
        model = build_pipeline()
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[{symbol}] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    out_path = os.path.join(MODELS_DIR, f"{symbol}_xgb.joblib")
    joblib.dump(model, out_path)
    print(f"[OK] Saved model: {out_path}")
    return out_path

def train_all(symbols: List[str] = None, do_search: bool = False):
    if symbols is None:
        # find all processed files
        files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith('_features.csv')]
        symbols = [f.replace('_features.csv','') for f in files]
    for s in symbols:
        try:
            train_symbol(s, do_search=do_search)
        except Exception as e:
            print(f"[ERROR] training {s}: {e}")

if __name__ == "__main__":
    train_all(do_search=False)
