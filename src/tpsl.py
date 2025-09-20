# src/tpsl.py
import numpy as np

def map_prob_to_tpsl(prob: float, k: float = 3.0, base_tp=0.01, base_sl=0.01, max_tp=0.08, min_sl=0.002, neutral_band=0.03):
    """
    Convert model probability (0..1) into trade direction and TP/SL percentages.
    - If prob around 0.5 (within neutral_band) -> return None (no trade)
    - Uses logistic-like mapping to scale TP and SL
    """
    if prob is None or np.isnan(prob):
        return None
    p = float(np.clip(prob, 1e-6, 1-1e-6))
    if abs(p - 0.5) <= neutral_band:
        return None
    direction = 'long' if p > 0.5 else 'short'
    # convert distance from 0.5 to confidence 0..1
    conf = (abs(p - 0.5) - neutral_band) / (0.5 - neutral_band)
    conf = np.clip(conf, 0.0, 1.0)
    # shape by exponent
    conf = 1/(1+np.exp(-k*(conf-0.5)))
    tp = base_tp + (max_tp - base_tp) * conf
    sl = base_sl - (base_sl - min_sl) * conf
    return {'direction': direction, 'tp_pct': float(tp), 'sl_pct': float(sl)}

