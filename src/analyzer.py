# analyzer.py
import os
import re
import logging
import pandas as pd
import numpy as np
from ta import add_all_ta_features
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== KONFIGURASI =====================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
TIMEFRAMES = ["1D", "4h"]                     # hanya 1D dan 4H
WEIGHT_TECHNICAL = 0.7
WEIGHT_FUNDAMENTAL = 0.3
MAX_DATA_ROWS = 500                           # batasi untuk efisiensi

# Bobot teknikal per timeframe (harus sesuai urutan)
TECH_WEIGHTS = {'1D': 0.6, '4h': 0.4}         # 1D 60%, 4H 40%

# ===================== FUNGSI LOAD DATA =====================
def load_data(pair, tf):
    """Membaca file CSV, mengembalikan DataFrame dengan maksimal MAX_DATA_ROWS baris terakhir."""
    filename = f"{pair}_{tf}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        logging.warning(f"File {filepath} tidak ditemukan.")
        return None

    try:
        df = pd.read_csv(filepath, sep='\t', header=None,
                         names=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'],
                         parse_dates=['Datetime'], dayfirst=False)
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)

        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            logging.warning(f"Data {filename} kosong setelah dibersihkan.")
            return None

        if len(df) > MAX_DATA_ROWS:
            df = df.iloc[-MAX_DATA_ROWS:]

        logging.info(f"Berhasil memuat {len(df)} baris dari {filename}")
        return df
    except Exception as e:
        logging.error(f"Gagal membaca {filepath}: {e}")
        return None

# ===================== INDIKATOR TEKNIKAL =====================
def add_indicators(df):
    """Menambahkan indikator, dengan pengecekan kolom minimal."""
    if df is None or df.empty:
        return df

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logging.error("Dataframe tidak memiliki kolom yang diperlukan")
        return df

    try:
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
    except Exception as e:
        logging.error(f"Error saat menambah indikator: {e}")
    return df

# ===================== DETEKSI SINYAL =====================
def detect_signals(df):
    """Deteksi sinyal, mengembalikan skor buy/sell dan alasan."""
    if df is None or df.empty:
        return 0, 0, ["Data tidak cukup"]

    last = df.iloc[-1]
    buy_score = 0
    sell_score = 0
    reasons = []

    # 1. Trend SMA
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        if pd.notna(last['sma_20']) and pd.notna(last['sma_50']):
            if last['Close'] > last['sma_20'] and last['sma_20'] > last['sma_50']:
                buy_score += 15
                reasons.append("Harga di atas SMA20 & SMA20 > SMA50 (uptrend)")
            elif last['Close'] < last['sma_20'] and last['sma_20'] < last['sma_50']:
                sell_score += 15
                reasons.append("Harga di bawah SMA20 & SMA20 < SMA50 (downtrend)")

    # 2. MACD
    if 'trend_macd' in df.columns and 'trend_macd_signal' in df.columns:
        if pd.notna(last['trend_macd']) and pd.notna(last['trend_macd_signal']):
            if last['trend_macd'] > last['trend_macd_signal'] and last['trend_macd'] > 0:
                buy_score += 10
                reasons.append("MACD di atas signal line (bullish)")
            elif last['trend_macd'] < last['trend_macd_signal'] and last['trend_macd'] < 0:
                sell_score += 10
                reasons.append("MACD di bawah signal line (bearish)")

    # 3. RSI
    if 'momentum_rsi' in df.columns:
        if pd.notna(last['momentum_rsi']):
            if last['momentum_rsi'] < 30:
                buy_score += 15
                reasons.append("RSI oversold (<30)")
            elif last['momentum_rsi'] > 70:
                sell_score += 15
                reasons.append("RSI overbought (>70)")
            elif last['momentum_rsi'] > 50:
                buy_score += 5
                reasons.append("RSI di atas 50")
            elif last['momentum_rsi'] < 50:
                sell_score += 5
                reasons.append("RSI di bawah 50")

    # 4. Stochastic
    if 'momentum_stoch' in df.columns and 'momentum_stoch_signal' in df.columns:
        if pd.notna(last['momentum_stoch']) and pd.notna(last['momentum_stoch_signal']):
            if last['momentum_stoch'] < 20 and last['momentum_stoch_signal'] < 20:
                buy_score += 10
                reasons.append("Stochastic oversold")
            elif last['momentum_stoch'] > 80 and last['momentum_stoch_signal'] > 80:
                sell_score += 10
                reasons.append("Stochastic overbought")

    # 5. Bollinger Bands
    if 'volatility_bbl' in df.columns and 'volatility_bbh' in df.columns:
        if pd.notna(last['volatility_bbl']) and pd.notna(last['volatility_bbh']):
            if last['Close'] < last['volatility_bbl']:
                buy_score += 10
                reasons.append("Harga menyentuh lower band (potensi rebound)")
            elif last['Close'] > last['volatility_bbh']:
                sell_score += 10
                reasons.append("Harga menyentuh upper band (potensi koreksi)")

    # 6. Support/Resistance
    if 'support' in df.columns:
        if pd.notna(last['support']) and last['Close'] <= last['support'] * 1.01:
            buy_score += 10
            reasons.append("Mendekati level support")
    if 'resistance' in df.columns:
        if pd.notna(last['resistance']) and last['Close'] >= last['resistance'] * 0.99:
            sell_score += 10
            reasons.append("Mendekati level resistance")

    max_score = 70
    buy_score = min(100, (buy_score / max_score) * 100)
    sell_score = min(100, (sell_score / max_score) * 100)

    return buy_score, sell_score, reasons

# ===================== ANALISIS MULTI-TIMEFRAME =====================
def multi_timeframe_analysis(data_dict):
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
            'last_close': float(df['Close'].iloc[-1]) if not df.empty else None,
            'atr': float(df['volatility_atr'].iloc[-1]) if ('volatility_atr' in df and not df.empty) else 0
        }
    return signals

def combine_technical_signals(signals):
    """Gabungkan skor teknikal dengan bobot yang sudah ditentukan."""
    total_buy = 0
    total_sell = 0
    for tf, sig in signals.items():
        if sig is None:
            continue
        total_buy += sig['buy_score'] * TECH_WEIGHTS[tf]
        total_sell += sig['sell_score'] * TECH_WEIGHTS[tf]

    if total_buy > total_sell:
        return "BUY", total_buy
    elif total_sell > total_buy:
        return "SELL", total_sell
    else:
        return "NEUTRAL", 50.0

# ===================== ANALISIS FUNDAMENTAL =====================
def extract_currency_sentiment(text):
    if not text or not text.strip():
        return {c: 0.0 for c in ['EUR', 'USD', 'GBP', 'JPY', 'CHF']}

    currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF']
    currency_pattern = r'\b(' + '|'.join(currencies) + r')\b'
    sentences = re.split(r'[.!?]', text)

    positive_words = ['baik', 'naik', 'meningkat', 'positif', 'kuat', 'tinggi',
                      'hawkish', 'rate hike', 'surplus', 'optimis', 'menguat']
    negative_words = ['buruk', 'turun', 'menurun', 'negatif', 'lemah', 'rendah',
                      'dovish', 'rate cut', 'defisit', 'resesi', 'melemah']

    currency_sentiment = {c: [] for c in currencies}

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        mentioned = set(re.findall(currency_pattern, sent))
        if not mentioned:
            continue

        score = 0.0
        lower_sent = sent.lower()
        for word in positive_words:
            if word in lower_sent:
                score += 0.2
        for word in negative_words:
            if word in lower_sent:
                score -= 0.2
        sentiment_score = max(-1.0, min(1.0, score))

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
    raw_score = diff * 50
    if raw_score > 0:
        return "BUY", raw_score
    elif raw_score < 0:
        return "SELL", -raw_score
    else:
        return "NEUTRAL", 50.0

# ===================== KOMBINASI =====================
def combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf):
    dir_val = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
    tech_val = dir_val[tech_dir] * tech_conf
    fund_val = dir_val[fund_dir] * fund_conf
    combined = WEIGHT_TECHNICAL * tech_val + WEIGHT_FUNDAMENTAL * fund_val
    if combined > 0:
        return "BUY", abs(combined)
    elif combined < 0:
        return "SELL", abs(combined)
    else:
        return "NEUTRAL", 50.0

def calculate_sl_tp(entry_price, atr, direction):
    if direction == "BUY":
        return entry_price - atr, entry_price + atr
    elif direction == "SELL":
        return entry_price + atr, entry_price - atr
    else:
        return None, None

# ===================== FUNGSI UTAMA =====================
def analyze_pair(pair, news_text):
    logging.info(f"Memulai analisis untuk {pair}")
    data = {}
    for tf in TIMEFRAMES:
        df = load_data(pair, tf)
        data[tf] = df

    signals = multi_timeframe_analysis(data)
    tech_dir, tech_conf = combine_technical_signals(signals)

    if news_text.strip():
        currency_sentiment = extract_currency_sentiment(news_text)
        fund_dir, fund_conf = evaluate_fundamental(pair, currency_sentiment)
    else:
        currency_sentiment = {}
        fund_dir, fund_conf = "NEUTRAL", 50.0

    final_dir, final_conf = combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf)

    # Entry, SL, TP dari timeframe terkecil (4h)
    df_entry = data.get('4h')
    entry_price = sl = tp = None
    if df_entry is not None and not df_entry.empty and signals.get('4h') is not None:
        last_row = df_entry.iloc[-1]
        entry_price = float(last_row['Close'])
        atr_entry = signals['4h'].get('atr', 0)
        if final_dir in ["BUY", "SELL"] and atr_entry > 0:
            sl, tp = calculate_sl_tp(entry_price, atr_entry, final_dir)

    # Data chart dari 4H
    chart_data = None
    if df_entry is not None and not df_entry.empty:
        df_plot = add_indicators(df_entry.copy()).reset_index()
        
        # Ganti NaN dengan None agar valid JSON
        def clean_series(series, decimals=5):
            return [None if pd.isna(x) else round(x, decimals) for x in series]
        
        chart_data = {
            'time': df_plot['Datetime'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'open': clean_series(df_plot['Open']),
            'high': clean_series(df_plot['High']),
            'low': clean_series(df_plot['Low']),
            'close': clean_series(df_plot['Close']),
            'volume': clean_series(df_plot['Volume'], decimals=0),
            'sma20': clean_series(df_plot['sma_20']),
            'sma50': clean_series(df_plot['sma_50']),
            'sma200': clean_series(df_plot['sma_200']),
            'bb_upper': clean_series(df_plot['volatility_bbh']),
            'bb_lower': clean_series(df_plot['volatility_bbl']),
            'rsi': clean_series(df_plot['momentum_rsi'], decimals=2),
            'macd': clean_series(df_plot['trend_macd']),
            'macd_signal': clean_series(df_plot['trend_macd_signal']),
            'macd_hist': clean_series(df_plot['trend_macd_diff'])
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
        'signals': {
            tf: {
                'buy_score': round(signals[tf]['buy_score'], 2) if signals[tf] else None,
                'sell_score': round(signals[tf]['sell_score'], 2) if signals[tf] else None,
                'last_close': round(signals[tf]['last_close'], 5) if signals[tf] else None,
                'reasons': signals[tf]['reasons'] if signals[tf] else []
            } for tf in TIMEFRAMES if signals.get(tf) is not None
        },
        'currency_sentiment': {k: round(v, 2) for k, v in currency_sentiment.items() if v != 0},
        'chart_data': chart_data
    }
    logging.info(f"Analisis selesai untuk {pair}")
    return result
