# analyzer.py
import os
import re
import logging
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from datetime import datetime, timedelta
from twelve_data import download_and_cache, get_intraday_data, get_real_time_price, get_quote
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ===================== KONFIGURASI =====================
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
# Timeframe untuk analisis multi-timeframe (bobot diberikan sesuai)
TIMEFRAMES = ["1D", "4h", "1h", "15min"]
WEIGHT_TECHNICAL = 0.7
WEIGHT_FUNDAMENTAL = 0.3
MAX_DATA_ROWS = 500
# Bobot untuk masing-masing timeframe (total harus 1.0)
TECH_WEIGHTS = {'1D': 0.3, '4h': 0.3, '1h': 0.25, '15min': 0.15}

# ===================== FUNGSI BANTU =====================
def ensure_columns(df):
    """
    Memastikan dataframe memiliki kolom yang diperlukan.
    Jika kolom 'Volume' tidak ada, tambahkan dengan nilai 0.
    """
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            if col == 'Volume':
                logging.warning("Kolom Volume tidak ditemukan, menambahkan dengan nilai 0.")
                df['Volume'] = 0
            else:
                logging.error(f"Kolom {col} tidak ditemukan! Data tidak dapat digunakan.")
                return None
    return df

# ===================== FUNGSI LOAD DATA (LOKAL) =====================
def load_data(pair, tf):
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
            return None

        if len(df) > MAX_DATA_ROWS:
            df = df.iloc[-MAX_DATA_ROWS:]

        return df
    except Exception as e:
        logging.error(f"Gagal membaca {filepath}: {e}")
        return None

# ===================== INDIKATOR TEKNIKAL =====================
def add_indicators(df):
    if df is None or df.empty:
        return df

    df = ensure_columns(df)
    if df is None:
        return None

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
        if df is None:
            signals[tf] = None
            continue
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

# ===================== FUNGSI UTAMA ANALISIS (LOKAL) =====================
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

    # Entry dari timeframe terkecil (15min) jika ada, fallback ke 1h
    df_entry = data.get('15min') or data.get('1h')
    entry_price = sl = tp = None
    if df_entry is not None and not df_entry.empty:
        # Untuk entry, kita menggunakan harga terakhir dari dataframe (tanpa real-time di sini)
        last_row = df_entry.iloc[-1]
        entry_price = float(last_row['Close'])
        # Cari sinyal untuk timeframe entry
        entry_tf = '15min' if '15min' in signals and signals['15min'] else '1h'
        atr_entry = signals[entry_tf]['atr'] if entry_tf in signals and signals[entry_tf] else 0
        if final_dir in ["BUY", "SELL"] and atr_entry > 0:
            sl, tp = calculate_sl_tp(entry_price, atr_entry, final_dir)

    # Data chart dari 1h (bisa disesuaikan)
    chart_data = None
    df_chart = data.get('1h')
    if df_chart is not None and not df_chart.empty:
        df_plot = add_indicators(df_chart.copy()).reset_index()
        if df_plot is not None:
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
    return result

# ===================== LIVE PREDICTION (dengan real-time dan M15) =====================
def get_live_prediction(pair):
    api_symbol = f"{pair[:3]}/{pair[3:]}"
    
    # Dapatkan quote dan harga real-time
    quote = get_quote(api_symbol)
    current_price = get_real_time_price(api_symbol)
    if current_price is None:
        return {"error": "Gagal mendapatkan harga real-time"}
    
    # Ambil data historis dari cache atau API untuk semua timeframe
    df_15min = download_and_cache(api_symbol, "15min")
    df_1h = download_and_cache(api_symbol, "1h")
    df_4h = download_and_cache(api_symbol, "4h")
    df_1d = download_and_cache(api_symbol, "1day")
    
    if any(x is None for x in [df_15min, df_1h, df_4h, df_1d]):
        return {"error": "Gagal mendapatkan data historis untuk salah satu timeframe"}
    
    # Pastikan kolom lengkap
    df_15min = ensure_columns(df_15min)
    df_1h = ensure_columns(df_1h)
    df_4h = ensure_columns(df_4h)
    df_1d = ensure_columns(df_1d)
    if any(x is None for x in [df_15min, df_1h, df_4h, df_1d]):
        return {"error": "Data historis tidak memiliki kolom yang diperlukan"}
    
    # Untuk M15, kita akan membuat candle terkini (incomplete) berdasarkan real-time price
    # Ambil candle terakhir yang sudah ditutup
    last_candle = df_15min.iloc[-1].copy()
    last_time = df_15min.index[-1]
    # Perkiraan candle baru: open = close sebelumnya, close = current_price, high = max(high sebelumnya, current), low = min(low sebelumnya, current)
    new_open = last_candle['Close']
    new_high = max(last_candle['High'], current_price)
    new_low = min(last_candle['Low'], current_price)
    new_close = current_price
    new_volume = 0  # volume tidak diketahui
    
    # Buat dataframe untuk candle baru
    new_candle = pd.DataFrame({
        'Open': [new_open],
        'High': [new_high],
        'Low': [new_low],
        'Close': [new_close],
        'Volume': [new_volume]
    }, index=[last_time + timedelta(minutes=15)])  # perkiraan timestamp berikutnya
    
    # Gabungkan dengan data historis
    df_15min_extended = pd.concat([df_15min, new_candle])
    
    # Sekarang siapkan dictionary data untuk semua timeframe
    data_dict = {
        '1D': df_1d,
        '4h': df_4h,
        '1h': df_1h,
        '15min': df_15min_extended
    }
    
    signals = multi_timeframe_analysis(data_dict)
    tech_dir, tech_conf = combine_technical_signals(signals)
    
    # Untuk SL/TP, kita gunakan ATR dari M15 (candle extended)
    atr_15min = signals['15min']['atr'] if signals['15min'] else 0
    if atr_15min > 0:
        sl, tp = calculate_sl_tp(current_price, atr_15min, tech_dir)
    else:
        sl = tp = None
    
    bid = float(quote.get('bid', current_price)) if quote and 'bid' in quote else current_price
    ask = float(quote.get('ask', current_price)) if quote and 'ask' in quote else current_price
    
    result = {
        'pair': pair,
        'current_price': round(current_price, 5),
        'bid': round(bid, 5),
        'ask': round(ask, 5),
        'timestamp': datetime.now().isoformat(),
        'prediction': {
            'direction': tech_dir,
            'confidence': round(tech_conf, 2),
            'entry': round(current_price, 5),  # entry menggunakan harga real-time
            'sl': round(sl, 5) if sl else None,
            'tp': round(tp, 5) if tp else None,
            'atr': round(atr_15min, 5)
        },
        'signals': {
            tf: {
                'buy_score': round(signals[tf]['buy_score'], 2) if signals[tf] else None,
                'sell_score': round(signals[tf]['sell_score'], 2) if signals[tf] else None,
                'reasons': signals[tf]['reasons'] if signals[tf] else []
            } for tf in TIMEFRAMES if signals.get(tf) is not None
        }
    }
    return result
