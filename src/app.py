import os
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta import add_all_ta_features
import warnings
warnings.filterwarnings('ignore')

# ===================== KONFIGURASI =====================
DATA_FOLDER = "python"          # folder tempat file CSV harga
PAIRS = ["EURUSD", "CHFJPY", "GBPJPY", "GBPUSD", "EURJPY"]
TIMEFRAMES = ["1D", "4h", "1H"]
RISK_REWARD = 1.0
MIN_PROBABILITY = 65            # minimal confidence untuk rekomendasi
WEIGHT_TECHNICAL = 0.7
WEIGHT_FUNDAMENTAL = 0.3

# ===================== INISIALISASI NLP =====================
try:
    from textblob import TextBlob
    USE_TEXTBLOB = True
except ImportError:
    USE_TEXTBLOB = False
    print("TextBlob tidak terinstall. Menggunakan aturan sentimen sederhana.")

# ===================== FUNGSI BANTU =====================
def load_data(pair, tf):
    """Membaca file CSV harga untuk pair dan timeframe tertentu."""
    filename = f"{pair}_{tf}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    if not os.path.exists(filepath):
        print(f"File {filepath} tidak ditemukan. Lewati.")
        return None

    df = pd.read_csv(filepath, parse_dates={'Datetime': ['Date', 'Time']},
                     index_col='Datetime', dayfirst=True)
    df.sort_index(inplace=True)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    return df

def add_indicators(df):
    """Menambahkan semua indikator teknikal ke dataframe."""
    df = add_all_ta_features(df, open="Open", high="High", low="Low",
                             close="Close", volume="Volume", fillna=True)
    # SMA tambahan
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    # EMA
    df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    # Support & Resistance dinamis
    window = 20
    df['resistance'] = df['High'].rolling(window=window, center=True).max()
    df['support'] = df['Low'].rolling(window=window, center=True).min()
    return df

def detect_signals(df, tf_name):
    """
    Mendeteksi sinyal beli/jual berdasarkan indikator.
    Mengembalikan skor beli (0-100), skor jual (0-100), dan daftar alasan.
    """
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

    # Normalisasi skor maksimum 100
    max_score = 70  # total maksimum komponen di atas
    buy_score = min(100, (buy_score / max_score) * 100)
    sell_score = min(100, (sell_score / max_score) * 100)

    return buy_score, sell_score, reasons

def multi_timeframe_analysis(data_dict):
    """Analisis teknikal multi-timeframe."""
    signals = {}
    for tf, df in data_dict.items():
        if df is None or df.empty:
            signals[tf] = None
            continue
        df = add_indicators(df)
        buy, sell, reasons = detect_signals(df, tf)
        signals[tf] = {
            'buy_score': buy,
            'sell_score': sell,
            'reasons': reasons,
            'last_close': df['Close'].iloc[-1],
            'atr': df['volatility_atr'].iloc[-1]
        }
    return signals

def combine_technical_signals(signals):
    """Gabungkan skor teknikal dari semua timeframe dengan bobot."""
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
def extract_currency_sentiment(text):
    """
    Mengekstrak sentimen per mata uang dari teks berita.
    Mengembalikan dictionary: { 'EUR': sentiment_score (-1..1), ... }
    sentiment_score >0 = positif, <0 = negatif, 0 = netral.
    """
    # Daftar mata uang yang relevan
    currencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF']
    # Pola untuk menemukan penyebutan mata uang (bisa dalam bentuk kode atau nama)
    currency_pattern = r'\b(' + '|'.join(currencies) + r')\b'
    # Split teks menjadi kalimat (sederhana)
    sentences = re.split(r'[.!?]', text)
    
    currency_sentiment = {c: [] for c in currencies}
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Cari mata uang yang disebut dalam kalimat ini
        mentioned = set(re.findall(currency_pattern, sent))
        if not mentioned:
            continue
        
        # Hitung sentimen kalimat
        if USE_TEXTBLOB:
            blob = TextBlob(sent)
            sentiment_score = blob.sentiment.polarity  # -1..1
        else:
            # Aturan sederhana: hitung kata positif/negatif
            positive_words = ['baik', 'naik', 'meningkat', 'positif', 'kuat', 'tinggi', 'hawkish', 'rate hike', 'surplus']
            negative_words = ['buruk', 'turun', 'menurun', 'negatif', 'lemah', 'rendah', 'dovish', 'rate cut', 'defisit']
            score = 0
            for word in positive_words:
                if word in sent.lower():
                    score += 0.2
            for word in negative_words:
                if word in sent.lower():
                    score -= 0.2
            sentiment_score = max(-1, min(1, score))  # batasi -1..1
        
        # Catat sentimen untuk setiap mata uang yang disebut
        for curr in mentioned:
            currency_sentiment[curr].append(sentiment_score)
    
    # Rata-rata sentimen per mata uang
    result = {}
    for curr, scores in currency_sentiment.items():
        if scores:
            result[curr] = np.mean(scores)
        else:
            result[curr] = 0.0  # netral
    return result

def evaluate_fundamental(pair, currency_sentiment):
    """
    Menghitung skor fundamental dan arah untuk pair berdasarkan sentimen mata uang.
    """
    base = pair[:3]
    quote = pair[3:]
    base_sent = currency_sentiment.get(base, 0.0)
    quote_sent = currency_sentiment.get(quote, 0.0)
    
    # Logika: sentimen positif untuk base => bullish (buy), positif untuk quote => bearish (sell)
    # Sentimen dianggap sebagai kekuatan permintaan mata uang.
    # Selisih sentimen: base_sent - quote_sent => positif -> beli, negatif -> jual.
    diff = base_sent - quote_sent
    
    # Konversi ke skor 0-100 dengan arah
    # diff berkisar -2..2, kita petakan ke -100..100
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
    
    # Pastikan dalam rentang 0-100
    confidence = min(100, max(0, confidence))
    return direction, confidence

# ===================== KOMBINASI TEKNIKAL + FUNDAMENTAL =====================
def combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf):
    """Gabungkan arah dan confidence dari teknikal dan fundamental."""
    # Mapping arah ke nilai numerik: BUY=1, SELL=-1, NEUTRAL=0
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

# ===================== MANAJEMEN RISIKO =====================
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

# ===================== VISUALISASI =====================
def plot_chart(pair, tf, df, signals, direction, entry, sl, tp):
    """Plot candlestick chart dengan indikator dan level entry."""
    df = df.copy()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.5, 0.2, 0.15, 0.15])

    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Price'), row=1, col=1)

    # SMA
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], line=dict(color='blue', width=1), name='SMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['sma_200'], line=dict(color='red', width=1), name='SMA 200'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['volatility_bbh'], line=dict(color='gray', width=1, dash='dash'), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['volatility_bbl'], line=dict(color='gray', width=1, dash='dash'), name='BB Lower'), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['momentum_rsi'], line=dict(color='purple'), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['trend_macd'], line=dict(color='blue'), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['trend_macd_signal'], line=dict(color='orange'), name='Signal'), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['trend_macd_diff'], name='Histogram'), row=3, col=1)

    # Volume
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=4, col=1)

    # Entry, SL, TP
    if entry and sl and tp:
        fig.add_hline(y=entry, line_dash="solid", line_color="blue", line_width=2,
                      annotation_text=f"Entry: {entry:.5f}", row=1, col=1)
        fig.add_hline(y=sl, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"SL: {sl:.5f}", row=1, col=1)
        fig.add_hline(y=tp, line_dash="dash", line_color="green", line_width=2,
                      annotation_text=f"TP: {tp:.5f}", row=1, col=1)

    fig.update_layout(title=f'{pair} - {tf} - Analisis Teknikal + Fundamental',
                      xaxis_rangeslider_visible=False,
                      template='plotly_dark')
    fig.show()

# ===================== PANDUAN FUNDAMENTAL =====================
def fundamental_info(pair):
    """Menampilkan data fundamental yang perlu diperhatikan."""
    info = {
        "EURUSD": "Eurozone / US: Suku bunga ECB & The Fed, NFP, CPI, PDB, PMI",
        "CHFJPY": "Swiss / Jepang: Suku bunga SNB & BoJ, CPI, PDB, Tankan, Neraca dagang",
        "GBPJPY": "Inggris / Jepang: Suku bunga BoE & BoJ, CPI, PDB, Tenaga kerja, Tankan",
        "GBPUSD": "Inggris / US: Suku bunga BoE & The Fed, NFP, CPI, PDB, PMI",
        "EURJPY": "Eurozone / Jepang: Suku bunga ECB & BoJ, CPI, PDB, PMI, Tankan"
    }
    print(f"\n=== Analisis Fundamental untuk {pair} ===")
    print(info.get(pair, "Data tidak tersedia"))
    print("Link sumber: https://www.forexfactory.com/calendar")

# ===================== MAIN PROGRAM =====================
def main():
    if not os.path.exists(DATA_FOLDER):
        print(f"Folder '{DATA_FOLDER}' tidak ditemukan. Buat folder dan tempatkan file CSV harga di dalamnya.")
        return

    # Input fundamental naratif
    print("\nMasukkan berita/analisis fundamental (dalam bahasa Indonesia/Inggris).")
    print("Contoh: 'ECB menaikkan suku bunga, ini positif untuk EUR.'")
    print("Ketik berita (akhiri dengan baris kosong):")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    news_text = "\n".join(lines)

    # Ekstrak sentimen fundamental
    if news_text.strip():
        currency_sentiment = extract_currency_sentiment(news_text)
        print("\nSentimen terdeteksi per mata uang:")
        for curr, score in currency_sentiment.items():
            if score != 0:
                sent_str = "Positif" if score > 0 else "Negatif"
                print(f"  {curr}: {sent_str} ({score:.2f})")
    else:
        currency_sentiment = {c: 0.0 for c in ['EUR','USD','GBP','JPY','CHF']}
        print("Tidak ada berita. Fundamental dianggap netral.")

    # Analisis untuk setiap pair
    for pair in PAIRS:
        print(f"\n{'='*60}")
        print(f"Analisis untuk {pair}")
        print('='*60)

        # Load data teknikal
        data = {}
        for tf in TIMEFRAMES:
            df = load_data(pair, tf)
            data[tf] = df

        signals = multi_timeframe_analysis(data)
        tech_dir, tech_conf = combine_technical_signals(signals)

        # Evaluasi fundamental untuk pair ini
        fund_dir, fund_conf = evaluate_fundamental(pair, currency_sentiment)

        # Gabungkan
        final_dir, final_conf = combine_technical_fundamental(tech_dir, tech_conf, fund_dir, fund_conf)

        # Ambil data 1H untuk entry
        df_1h = data.get('1H')
        if df_1h is not None and not df_1h.empty:
            last_row = df_1h.iloc[-1]
            atr_1h = signals['1H']['atr'] if signals['1H'] else None
            entry_price = last_row['Close']
            if final_dir in ["BUY", "SELL"] and atr_1h is not None:
                sl, tp = calculate_sl_tp(entry_price, atr_1h, final_dir)
            else:
                sl = tp = None
        else:
            entry_price = sl = tp = None

        # Tampilkan hasil
        print(f"\n=== HASIL ANALISIS GABUNGAN ===")
        print(f"Arah: {final_dir}")
        print(f"Keyakinan: {final_conf:.2f}%")
        print(f"(Teknikal: {tech_dir} {tech_conf:.2f}%, Fundamental: {fund_dir} {fund_conf:.2f}%)")

        if final_dir != "NEUTRAL" and final_conf >= MIN_PROBABILITY:
            print(f"Harga Entry (1H): {entry_price:.5f}")
            print(f"Stop Loss: {sl:.5f}")
            print(f"Take Profit: {tp:.5f}")
            print("Risk-Reward: 1:1")
        else:
            print("Sinyal tidak cukup kuat (<65% atau netral). Tidak ada rekomendasi entry.")

        # Tampilkan rincian teknikal per timeframe
        for tf, sig in signals.items():
            if sig is None:
                continue
            print(f"\n--- Timeframe {tf} ---")
            print(f"  Buy Score: {sig['buy_score']:.2f} | Sell Score: {sig['sell_score']:.2f}")
            print(f"  Harga Terakhir: {sig['last_close']:.5f}")
            print(f"  Alasan: {', '.join(sig['reasons'][:3])}")

        # Plot chart 1H
        if df_1h is not None and not df_1h.empty:
            df_1h_plot = add_indicators(df_1h.copy())
            plot_chart(pair, '1H', df_1h_plot, signals['1H'], final_dir, entry_price, sl, tp)

        fundamental_info(pair)

        input("\nTekan Enter untuk melanjutkan ke pair berikutnya...")

if __name__ == "__main__":
    main()
