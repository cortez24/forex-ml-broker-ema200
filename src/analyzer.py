# analyzer.py
import os
import re
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
TIMEFRAMES = ["1D", "4h", "1H"]
WEIGHT_TECHNICAL = 0.7
WEIGHT_FUNDAMENTAL = 0.3

# ===================== FUNGSI BANTU =====================
def load_data(pair, tf):
    """Membaca file CSV tab-delimited tanpa header untuk pair dan timeframe tertentu."""
    filename = f"{pair}_{tf}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"File {filepath} tidak ditemukan.")
        return None

    # Baca CSV tanpa header, delimiter tab
    df = pd.read_csv(filepath, sep='\t', header=None,
                     names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                     parse_dates=['Datetime'], dayfirst=False)  # format tanggal 2010-02-09 sudah ISO
    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    # Konversi ke numerik
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    """Menambahkan semua indikator teknikal."""
    df = add_all_ta_features(df, open="Open", high="High", low="Low",
                             close="Close", volume="Volume", fillna=True)
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    window = 20
    df['resistance'] = df['High'].rolling(window=window, center=True).max()
    df['support'] = df['Low'].rolling(window=window, center=True).min()
    return df

def detect_signals(df):
    """Deteksi sinyal teknikal, kembalikan skor beli/jual dan alasan."""
    last = df.iloc[-1]
    buy_score = 0
    sell_score = 0
    reasons = []

    # 1. Trend SMA
    if last['Close'] > last['sma_20'] and last['sma_20'] > last['sma_50']:
        buy_score += 15
        reasons.append("Harga di atas SMA20 & SMA20 > SMA50 (uptrend)")
    elif last['Close'] < last['sma_20'] and last['sma_20'] < last['sma_50']:
        sell_score += 15
        reasons.append("Harga di bawah SMA20 & SMA20 < SMA50 (downtrend)")

    # 2. MACD
    if last['trend_macd'] > last['trend_macd_signal'] and last['trend_macd'] > 0:
        buy_score += 10
        reasons.append("MACD di atas signal line (bullish)")
    elif last['trend_macd'] < last['trend_macd_signal'] and last['trend_macd'] < 0:
        sell_score += 10
        reasons.append("MACD di bawah signal line (bearish)")

    # 3. RSI
    if last['momentum_rsi'] < 30:
        buy_score += 15
        reasons.append("RSI oversold (<30)")
    elif last['momentum_rsi'] > 70:
        sell_score += 15
        reasons.append("RSI overbought (>70)")
    elif last['momentum_rsi'] > 50:
        buy_score += 5
    elif last['momentum_rsi'] < 50:
        sell_score += 5

    # 4. Stochastic
    if last['momentum_stoch'] < 20 and last['momentum_stoch_signal'] < 20:
        buy_score += 10
        reasons.append("Stochastic oversold")
    elif last['momentum_stoch'] > 80 and last['momentum_stoch_signal'] > 80:
        sell_score += 10
        reasons.append("Stochastic overbought")

    # 5. Bollinger Bands
    if last['Close'] < last['volatility_bbl']:
        buy_score += 10
        reasons.append("Harga menyentuh lower band (potensi rebound)")
    elif last['Close'] > last['volatility_bbh']:
        sell_score += 10
        reasons.append("Harga menyentuh upper band (potensi koreksi)")

    # 6. Support/Resistance
    if last['Close'] <= last['support'] * 1.01:
        buy_score += 10
        reasons.append("Mendekati level support")
    if last['Close'] >= last['resistance'] * 0.99:
        sell_score += 10
        reasons.append("Mendekati level resistance")

    max_score = 70
    buy_score = min(100, (buy_score / max_score) * 100)
    sell_score = min(100, (sell_score / max_score) * 100)
    return buy_score, sell_score, reasons

def multi_timeframe_analysis(data_dict):
    """Analisis semua timeframe, kembalikan dict sinyal."""
    signals = {}
    for tf, df in data_dict.items():
        if df is None or df.empty:
            signals[tf] = None
            continue
        df = add_indicators(df)
        buy, sell, reasons = detect_signals(df)
        signals[tf] = {
            'buy_score': buy,
            'sell_score': sell,
            'reasons': reasons,
            'last_close': float(df['Close'].iloc[-1]),
            'atr': float(df['volatility_atr'].iloc[-1]) if 'volatility_atr' in df else 0
        }
    return signals

def combine_technical_signals(signals):
    """Gabungkan skor teknikal dengan bobot."""
    weights = {'1D': 0.4, '4h': 0.35, '1H': 0.25}
    total_buy = 0
    total_sell = 0
    for tf, sig in signals.items():
        if sig is None:
            continue
        total_buy += sig['buy_score'] * weights[tf]
        total_sell += sig['sell_score'] * weights[tf]

    if total_buy > total_sell:
        direction = "BUY"
        confidence = total_buy
    elif total_sell > total_buy:
        direction = "SELL"
        confidence = total_sell
    else:
        direction = "NEUTRAL"
        confidence = 50
    return direction, confidence

# ===================== ANALISIS FUNDAMENTAL NLP =====================
# (Menggunakan aturan sederhana, tanpa TextBlob agar tidak dependensi tambahan)
def extract_currency_sentiment(text):
    """Ekstrak sentimen per mata uang dari teks."""
    currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF']
    currency_pattern = r'\b(' + '|'.join(currencies) + r')\b'
    sentences = re.split(r'[.!?]', text)
    
    currency_sentiment = {c: [] for c in currencies}
    positive_words = ['baik', 'naik', 'meningkat', 'positif', 'kuat', 'tinggi', 'hawkish', 'rate hike', 'surplus']
    negative_words = ['buruk', 'turun', 'menurun', 'negatif', 'lemah', 'rendah', 'dovish', 'rate cut', 'defisit']
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        mentioned = set(re.findall(currency_pattern, sent))
        if not mentioned:
            continue
        
        # Hitung skor sederhana
        score = 0
        lower_sent = sent.lower()
        for word in positive_words:
            if word in lower_sent:
                score += 0.2
        for word in negative_words:
            if word in lower_sent:
                score -= 0.2
        sentiment_score = max(-1, min(1, score))
        
        for curr in mentioned:
            currency_sentiment[curr].append(sentiment_score)
    
    result = {}
    for curr, scores in currency_sentiment.items():
        result[curr] = np.mean(scores) if scores else 0.0
    return result

def evaluate_fundamental(pair, currency_sentiment):
    base = pair[:3]
    quote = pair[3:]
    base_sent = currency_sentiment.get(base, 0.0)
    quote_sent = currency_sentiment.get(quote, 0.0)
    
    diff = base_sent - quote_sent
    raw_score = diff * 50  # -100..100
    if raw_score > 0:
        direction = "BUY"
        confidence = raw_score
    elif raw_score < 0:
        direction = "SELL"
        confidence = -raw_score
    else:
        direction = "NEUTRAL"
        confidence = 50
    confidence = min(100, max(0, confidence))
    return direction, confidence

# ===================== KOMBINASI =====================
def combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf):
    dir_val = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
    tech_val = dir_val[tech_dir] * tech_conf
    fund_val = dir_val[fund_dir] * fund_conf
    combined = WEIGHT_TECHNICAL * tech_val + WEIGHT_FUNDAMENTAL * fund_val
    if combined > 0:
        final_dir = "BUY"
    elif combined < 0:
        final_dir = "SELL"
    else:
        final_dir = "NEUTRAL"
    final_conf = abs(combined)
    final_conf = min(100, final_conf)
    return final_dir, final_conf

def calculate_sl_tp(entry_price, atr, direction):
    if direction == "BUY":
        sl = entry_price - atr
        tp = entry_price + atr
    elif direction == "SELL":
        sl = entry_price + atr
        tp = entry_price - atr
    else:
        sl = tp = None
    return sl, tp

# ===================== FUNGSI UTAMA ANALISIS =====================
def analyze_pair(pair, news_text):
    # Load data teknikal
    data = {}
    for tf in TIMEFRAMES:
        df = load_data(pair, tf)
        data[tf] = df

    signals = multi_timeframe_analysis(data)
    tech_dir, tech_conf = combine_technical_signals(signals)

    # Fundamental
    if news_text.strip():
        currency_sentiment = extract_currency_sentiment(news_text)
        fund_dir, fund_conf = evaluate_fundamental(pair, currency_sentiment)
    else:
        fund_dir, fund_conf = "NEUTRAL", 50
        currency_sentiment = {}

    final_dir, final_conf = combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf)

    # Ambil data 1H untuk entry
    df_1h = data.get('1H')
    if df_1h is not None and not df_1h.empty and signals['1H'] is not None:
        last_row = df_1h.iloc[-1]
        entry_price = float(last_row['Close'])
        atr_1h = signals['1H']['atr']
        if final_dir in ["BUY", "SELL"] and atr_1h > 0:
            sl, tp = calculate_sl_tp(entry_price, atr_1h, final_dir)
        else:
            sl = tp = None
    else:
        entry_price = sl = tp = None

    # Siapkan data untuk chart (timeframe 1H)
    chart_data = None
    if df_1h is not None and not df_1h.empty:
        df_1h_plot = add_indicators(df_1h.copy()).reset_index()
        # Konversi ke format yang mudah di-frontend
        chart_data = {
            'time': df_1h_plot['Datetime'].astype(str).tolist(),
            'open': df_1h_plot['Open'].tolist(),
            'high': df_1h_plot['High'].tolist(),
            'low': df_1h_plot['Low'].tolist(),
            'close': df_1h_plot['Close'].tolist(),
            'volume': df_1h_plot['Volume'].tolist(),
            'sma20': df_1h_plot['sma_20'].tolist(),
            'sma50': df_1h_plot['sma_50'].tolist(),
            'sma200': df_1h_plot['sma_200'].tolist(),
            'bb_upper': df_1h_plot['volatility_bbh'].tolist(),
            'bb_lower': df_1h_plot['volatility_bbl'].tolist(),
            'rsi': df_1h_plot['momentum_rsi'].tolist(),
            'macd': df_1h_plot['trend_macd'].tolist(),
            'macd_signal': df_1h_plot['trend_macd_signal'].tolist(),
            'macd_hist': df_1h_plot['trend_macd_diff'].tolist()
        }

    result = {
        'pair': pair,
        'final_direction': final_dir,
        'final_confidence': round(final_conf, 2),
        'technical': {'direction': tech_dir, 'confidence': round(tech_conf, 2)},
        'fundamental': {'direction': fund_dir, 'confidence': round(fund_conf, 2)},
        'entry': round(entry_price, 5) if entry_price else None,
        'sl': round(sl, 5) if sl else None,
        'tp': round(tp, 5) if tp else None,
        'signals': signals,
        'chart_data': chart_data,
        'currency_sentiment': currency_sentiment
    }
    return result
