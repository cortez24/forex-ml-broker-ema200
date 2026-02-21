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
TIMEFRAMES = ["1D", "4h", "15min"]                     # tiga timeframe
WEIGHT_TECHNICAL = 0.7
WEIGHT_FUNDAMENTAL = 0.3
MAX_DATA_ROWS = 500
TECH_WEIGHTS = {'1D': 0.5, '4h': 0.3, '15min': 0.2}   # bobot sesuai prioritas

# ===================== FUNGSI BANTU =====================
def ensure_columns(df):
    """
    Memastikan dataframe memiliki kolom yang diperlukan untuk analisis teknikal.
    Jika kolom 'Volume' tidak ada, tambahkan dengan nilai 0.
    """
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            if col == 'Volume':
                logging.warning("Kolom Volume tidak ditemukan, menambahkan dengan nilai 0.")
                df['Volume'] = 0
            else:
                # Kolom harga harus ada, jika tidak maka dataframe tidak valid
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

    # Entry dari timeframe terkecil (15min)
    df_entry = data.get('15min')
    entry_price = sl = tp = None
    if df_entry is not None and not df_entry.empty and signals.get('15min') is not None:
        last_row = df_entry.iloc[-1]
        entry_price = float(last_row['Close'])
        atr_entry = signals['15min'].get('atr', 0)
        if final_dir in ["BUY", "SELL"] and atr_entry > 0:
            sl, tp = calculate_sl_tp(entry_price, atr_entry, final_dir)

    # Data chart dari 15min (untuk tampilan)
    chart_data = None
    if df_entry is not None and not df_entry.empty:
        df_plot = add_indicators(df_entry.copy()).reset_index()
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

# ===================== BACKTEST LOKAL =====================
def run_backtest(pair, start_date, end_date, min_confidence=65, max_hold_candles=20):
    logging.info(f"Memulai backtest lokal untuk {pair} dari {start_date} hingga {end_date}")

    data_1d = load_data(pair, "1D")
    data_4h = load_data(pair, "4h")
    data_15min = load_data(pair, "15min")

    if data_1d is None or data_4h is None or data_15min is None:
        return {"error": "Data tidak lengkap (1D, 4h, 15min)"}

    data_1d = data_1d.sort_index()
    data_4h = data_4h.sort_index()
    data_15min = data_15min.sort_index()

    data_1d = data_1d.loc[start_date:end_date]
    data_4h = data_4h.loc[start_date:end_date]
    data_15min = data_15min.loc[start_date:end_date]

    if data_15min.empty:
        return {"error": "Tidak ada data 15min dalam rentang"}

    min_history = 200
    if len(data_15min) < min_history + 1:
        return {"error": f"Data 15min terlalu sedikit, butuh minimal {min_history+1} candle"}

    trades = []

    # Kita perlu menyelaraskan sinyal pada timeframe 15min, dengan data 4h dan 1D yang sesuai pada saat itu.
    # Asumsikan kita loop pada data 15min, dan untuk setiap candle, kita ambil data 1D dan 4h yang tersedia hingga waktu tersebut.
    for i in range(min_history, len(data_15min)):
        df_15_current = data_15min.iloc[:i+1].copy()
        current_time = df_15_current.index[-1]

        df_4h_current = data_4h[data_4h.index <= current_time].copy()
        df_1d_current = data_1d[data_1d.index <= current_time].copy()

        if df_4h_current.empty or df_1d_current.empty:
            continue

        data_dict = {'1D': df_1d_current, '4h': df_4h_current, '15min': df_15_current}
        signals = multi_timeframe_analysis(data_dict)
        tech_dir, tech_conf = combine_technical_signals(signals)
        fund_dir, fund_conf = "NEUTRAL", 50.0
        final_dir, final_conf = combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf)

        if final_conf >= min_confidence and final_dir != "NEUTRAL":
            entry_price = df_15_current['Close'].iloc[-1]
            atr = signals['15min']['atr'] if signals['15min'] else 0
            if atr == 0:
                continue
            sl, tp = calculate_sl_tp(entry_price, atr, final_dir)

            exit_idx = None
            win = None
            # Cari exit dalam max_hold_candles candle 15min ke depan
            for j in range(i+1, min(i+1+max_hold_candles, len(data_15min))):
                candle = data_15min.iloc[j]
                if final_dir == "BUY":
                    if candle['High'] >= tp:
                        win = True
                        exit_idx = j
                        break
                    elif candle['Low'] <= sl:
                        win = False
                        exit_idx = j
                        break
                else:  # SELL
                    if candle['Low'] <= tp:
                        win = True
                        exit_idx = j
                        break
                    elif candle['High'] >= sl:
                        win = False
                        exit_idx = j
                        break

            if exit_idx is not None:
                if win:
                    profit_pct = (tp - entry_price) / entry_price if final_dir == "BUY" else (entry_price - tp) / entry_price
                else:
                    profit_pct = (sl - entry_price) / entry_price if final_dir == "BUY" else (entry_price - sl) / entry_price

                trades.append({
                    'entry_time': str(df_15_current.index[-1]),
                    'direction': final_dir,
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'exit_time': str(data_15min.index[exit_idx]),
                    'win': win,
                    'profit_pct': profit_pct * 100,
                    'confidence': final_conf
                })

    total_trades = len(trades)
    if total_trades == 0:
        return {"message": "Tidak ada sinyal yang memenuhi syarat"}

    wins = sum(1 for t in trades if t['win'])
    losses = total_trades - wins
    win_rate = wins / total_trades * 100
    total_profit_pct = sum(t['profit_pct'] for t in trades)
    avg_profit_pct = total_profit_pct / total_trades

    total_profit = sum(t['profit_pct'] for t in trades if t['win'])
    total_loss = abs(sum(t['profit_pct'] for t in trades if not t['win']))
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

    equity = 0
    peak = 0
    max_dd = 0
    for t in trades:
        equity += t['profit_pct']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / 100
        if dd > max_dd:
            max_dd = dd

    result = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 2),
        'total_profit_pct': round(total_profit_pct, 2),
        'avg_profit_pct': round(avg_profit_pct, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'trades': trades[-20:]
    }
    return result

# ===================== BACKTEST REAL-TIME (API) =====================
def run_backtest_realtime(pair, interval="15min", days_back=30, min_confidence=65):
    logging.info(f"Memulai backtest real-time untuk {pair} {interval} {days_back} days")

    api_symbol = f"{pair[:3]}/{pair[3:]}"
    # Data untuk timeframe entry (misal 15min)
    df_entry = get_intraday_data(api_symbol, interval, days=days_back)

    if df_entry is None or df_entry.empty:
        return {"error": "Gagal mendapatkan data dari API"}

    df_entry = ensure_columns(df_entry)
    if df_entry is None:
        return {"error": "Data entry tidak memiliki kolom yang diperlukan"}

    # Dapatkan juga data 1D dan 4h untuk analisis multi-timeframe
    df_4h = download_and_cache(api_symbol, "4h")
    df_1d = download_and_cache(api_symbol, "1day")

    if df_4h is None or df_1d is None:
        return {"error": "Gagal mendapatkan data 4H atau 1D"}

    df_4h = ensure_columns(df_4h)
    df_1d = ensure_columns(df_1d)
    if df_4h is None or df_1d is None:
        return {"error": "Data 4H/1D tidak memiliki kolom yang diperlukan"}

    # Resample data entry menjadi 4h dan 1d jika perlu? Tidak, kita gunakan yang sudah ada.

    min_history = 200
    if len(df_entry) < min_history + 1:
        return {"error": f"Data terlalu sedikit, butuh minimal {min_history+1} candle"}

    trades = []

    for i in range(min_history, len(df_entry)):
        df_entry_current = df_entry.iloc[:i+1].copy()
        current_time = df_entry_current.index[-1]

        df_4h_current = df_4h[df_4h.index <= current_time].copy()
        df_1d_current = df_1d[df_1d.index <= current_time].copy()

        if df_4h_current.empty or df_1d_current.empty:
            continue

        data_dict = {'1D': df_1d_current, '4h': df_4h_current, '15min': df_entry_current}
        signals = multi_timeframe_analysis(data_dict)
        tech_dir, tech_conf = combine_technical_signals(signals)
        fund_dir, fund_conf = "NEUTRAL", 50.0
        final_dir, final_conf = combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf)

        if final_conf >= min_confidence and final_dir != "NEUTRAL":
            entry_price = df_entry_current['Close'].iloc[-1]
            atr = signals['15min']['atr'] if signals['15min'] else 0
            if atr == 0:
                continue
            sl, tp = calculate_sl_tp(entry_price, atr, final_dir)

            max_hold = 20
            exit_idx = None
            win = None

            for j in range(i+1, min(i+1+max_hold, len(df_entry))):
                candle = df_entry.iloc[j]
                if final_dir == "BUY":
                    if candle['High'] >= tp:
                        win = True
                        exit_idx = j
                        break
                    elif candle['Low'] <= sl:
                        win = False
                        exit_idx = j
                        break
                else:
                    if candle['Low'] <= tp:
                        win = True
                        exit_idx = j
                        break
                    elif candle['High'] >= sl:
                        win = False
                        exit_idx = j
                        break

            if exit_idx is not None:
                if win:
                    profit_pct = (tp - entry_price) / entry_price if final_dir == "BUY" else (entry_price - tp) / entry_price
                else:
                    profit_pct = (sl - entry_price) / entry_price if final_dir == "BUY" else (entry_price - sl) / entry_price

                trades.append({
                    'entry_time': str(df_entry_current.index[-1]),
                    'direction': final_dir,
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'exit_time': str(df_entry.index[exit_idx]),
                    'win': win,
                    'profit_pct': profit_pct * 100,
                    'confidence': final_conf
                })

    total_trades = len(trades)
    if total_trades == 0:
        return {"message": "Tidak ada sinyal yang memenuhi syarat"}

    wins = sum(1 for t in trades if t['win'])
    losses = total_trades - wins
    win_rate = wins / total_trades * 100
    total_profit_pct = sum(t['profit_pct'] for t in trades)
    avg_profit_pct = total_profit_pct / total_trades

    total_profit = sum(t['profit_pct'] for t in trades if t['win'])
    total_loss = abs(sum(t['profit_pct'] for t in trades if not t['win']))
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

    equity = 0
    peak = 0
    max_dd = 0
    for t in trades:
        equity += t['profit_pct']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / 100
        if dd > max_dd:
            max_dd = dd

    result = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 2),
        'total_profit_pct': round(total_profit_pct, 2),
        'avg_profit_pct': round(avg_profit_pct, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'trades': trades[-20:],
        'data_period': f"{df_entry.index[0]} to {df_entry.index[-1]}"
    }
    return result

# ===================== LIVE PREDICTION =====================
def get_live_prediction(pair):
    api_symbol = f"{pair[:3]}/{pair[3:]}"

    quote = get_quote(api_symbol)
    current_price = get_real_time_price(api_symbol)

    if current_price is None:
        return {"error": "Gagal mendapatkan harga real-time"}

    # Dapatkan data historis dari cache atau API
    df_15min = download_and_cache(api_symbol, "15min")
    df_4h = download_and_cache(api_symbol, "4h")
    df_1d = download_and_cache(api_symbol, "1day")

    if df_15min is None or df_4h is None or df_1d is None:
        return {"error": "Gagal mendapatkan data historis (15min, 4H, 1D)"}

    df_15min = ensure_columns(df_15min)
    df_4h = ensure_columns(df_4h)
    df_1d = ensure_columns(df_1d)
    if df_15min is None or df_4h is None or df_1d is None:
        return {"error": "Data historis tidak memiliki kolom yang diperlukan"}

    data_dict = {'1D': df_1d, '4h': df_4h, '15min': df_15min}
    signals = multi_timeframe_analysis(data_dict)
    tech_dir, tech_conf = combine_technical_signals(signals)

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
            'entry': round(current_price, 5),
            'sl': round(sl, 5) if sl else None,
            'tp': round(tp, 5) if tp else None,
            'atr': round(atr_15min, 5)
        },
        'signals': {
            tf: {
                'buy_score': round(signals[tf]['buy_score'], 2) if signals[tf] else None,
                'sell_score': round(signals[tf]['sell_score'], 2) if signals[tf] else None,
                'reasons': signals[tf]['reasons'] if signals[tf] else []
            } for tf in ['1D', '4h', '15min'] if signals.get(tf) is not None
        }
    }
    return result
