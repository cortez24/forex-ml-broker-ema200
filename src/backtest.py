# src/backtest.py
import pandas as pd
import numpy as np
from tpsl import map_prob_to_tpsl
from typing import List, Dict

def simulate(price_df: pd.DataFrame, probs: pd.Series, max_holding: int = 48):
    """
    price_df: must contain columns time, open, high, low, close
    probs: index aligned with price_df (probability of UP)
    """
    trades = []
    equity = [1.0]  # normalized
    for i in range(len(price_df)-1):
        p = probs.iloc[i]
        if p is None or np.isnan(p):
            equity.append(equity[-1]); continue
        mapping = map_prob_to_tpsl(p)
        if mapping is None:
            equity.append(equity[-1]); continue
        direction = mapping['direction']
        tp_pct = mapping['tp_pct']; sl_pct = mapping['sl_pct']
        entry = price_df['close'].iloc[i]
        exit_price = None; reason=None
        for j in range(i+1, min(len(price_df), i+1+max_holding)):
            hi = price_df['high'].iloc[j]; lo = price_df['low'].iloc[j]
            if direction == 'long':
                if lo <= entry*(1-sl_pct):
                    exit_price = entry*(1-sl_pct); reason='SL'; break
                if hi >= entry*(1+tp_pct):
                    exit_price = entry*(1+tp_pct); reason='TP'; break
            else:
                if hi >= entry*(1+sl_pct):
                    exit_price = entry*(1+sl_pct); reason='SL'; break
                if lo <= entry*(1-tp_pct):
                    exit_price = entry*(1-tp_pct); reason='TP'; break
        if exit_price is None:
            j = min(len(price_df)-1, i+max_holding)
            exit_price = price_df['close'].iloc[j]; reason='Timeout'
        if direction == 'long':
            ret = (exit_price/entry)-1
        else:
            ret = (entry/exit_price)-1
        trades.append({'entry_idx':i, 'exit_idx':j, 'entry_price':entry, 'exit_price':exit_price, 'return':ret, 'reason':reason})
        equity.append(equity[-1]*(1+ret))
    return pd.DataFrame(trades), pd.Series(equity)

# Example usage: supply probs from model.predict_proba(features)[:,1]

