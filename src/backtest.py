import pandas as pd
import numpy as np
from analyzer import load_data, add_indicators, detect_signals, combine_technical_signals, evaluate_fundamental, combine_technical_fundamental, calculate_sl_tp
import logging

logging.basicConfig(level=logging.INFO)

def backtest_pair(pair, start_date=None, end_date=None, initial_balance=10000):
    # Load data
    df_1d = load_data(pair, '1D')
    df_4h = load_data(pair, '4h')
    
    if df_1d is None or df_4h is None:
        logging.error(f"Data untuk {pair} tidak lengkap.")
        return
    
    # Tambah indikator ke kedua dataframe
    df_1d = add_indicators(df_1d)
    df_4h = add_indicators(df_4h)
    
    # Pastikan indeks datetime
    df_1d.sort_index(inplace=True)
    df_4h.sort_index(inplace=True)
    
    # Filter tanggal jika perlu
    if start_date:
        df_1d = df_1d[df_1d.index >= start_date]
        df_4h = df_4h[df_4h.index >= start_date]
    if end_date:
        df_1d = df_1d[df_1d.index <= end_date]
        df_4h = df_4h[df_4h.index <= end_date]
    
    # Kita akan iterate pada df_4h
    trades = []
    balance = initial_balance
    in_position = False
    entry_price = None
    sl = None
    tp = None
    entry_time = None
    direction = None
    
    # Bobot teknikal (sama seperti di analyzer)
    tech_weights = {'1D': 0.6, '4h': 0.4}
    
    # Untuk setiap baris di df_4h (mulai dari indeks ke-200 agar indikator stabil)
    min_index = max(200, df_4h.shape[0] // 2)  # sederhana, pastikan indikator tidak NaN
    for i in range(min_index, len(df_4h)):
        current_time = df_4h.index[i]
        
        # Ambil data 1D hingga waktu ini
        df_1d_up_to = df_1d[df_1d.index <= current_time]
        if df_1d_up_to.empty:
            continue
        # Ambil baris terakhir dari 1D
        row_1d = df_1d_up_to.iloc[-1]
        # Ambil baris saat ini dari 4h
        row_4h = df_4h.iloc[i]
        
        # Hitung skor teknikal untuk masing-masing timeframe menggunakan baris tersebut
        # Kita perlu fungsi yang menghitung skor dari satu baris (dengan nilai indikator)
        # Definisikan fungsi di dalam atau gunakan fungsi baru
        
        def calculate_scores_from_row(row):
            # Ini adalah replika dari detect_signals tetapi untuk satu baris
            buy_score = 0
            sell_score = 0
            reasons = []
            
            # 1. SMA
            if pd.notna(row.get('sma_20')) and pd.notna(row.get('sma_50')):
                if row['Close'] > row['sma_20'] and row['sma_20'] > row['sma_50']:
                    buy_score += 15
                    reasons.append("Harga di atas SMA20 & SMA20 > SMA50 (uptrend)")
                elif row['Close'] < row['sma_20'] and row['sma_20'] < row['sma_50']:
                    sell_score += 15
                    reasons.append("Harga di bawah SMA20 & SMA20 < SMA50 (downtrend)")
            
            # 2. MACD
            if pd.notna(row.get('trend_macd')) and pd.notna(row.get('trend_macd_signal')):
                if row['trend_macd'] > row['trend_macd_signal'] and row['trend_macd'] > 0:
                    buy_score += 10
                    reasons.append("MACD di atas signal line (bullish)")
                elif row['trend_macd'] < row['trend_macd_signal'] and row['trend_macd'] < 0:
                    sell_score += 10
                    reasons.append("MACD di bawah signal line (bearish)")
            
            # 3. RSI
            if pd.notna(row.get('momentum_rsi')):
                if row['momentum_rsi'] < 30:
                    buy_score += 15
                    reasons.append("RSI oversold (<30)")
                elif row['momentum_rsi'] > 70:
                    sell_score += 15
                    reasons.append("RSI overbought (>70)")
                elif row['momentum_rsi'] > 50:
                    buy_score += 5
                    reasons.append("RSI di atas 50")
                elif row['momentum_rsi'] < 50:
                    sell_score += 5
                    reasons.append("RSI di bawah 50")
            
            # 4. Stochastic
            if pd.notna(row.get('momentum_stoch')) and pd.notna(row.get('momentum_stoch_signal')):
                if row['momentum_stoch'] < 20 and row['momentum_stoch_signal'] < 20:
                    buy_score += 10
                    reasons.append("Stochastic oversold")
                elif row['momentum_stoch'] > 80 and row['momentum_stoch_signal'] > 80:
                    sell_score += 10
                    reasons.append("Stochastic overbought")
            
            # 5. Bollinger Bands
            if pd.notna(row.get('volatility_bbl')) and pd.notna(row.get('volatility_bbh')):
                if row['Close'] < row['volatility_bbl']:
                    buy_score += 10
                    reasons.append("Harga menyentuh lower band (potensi rebound)")
                elif row['Close'] > row['volatility_bbh']:
                    sell_score += 10
                    reasons.append("Harga menyentuh upper band (potensi koreksi)")
            
            # 6. Support/Resistance
            if pd.notna(row.get('support')) and row['Close'] <= row['support'] * 1.01:
                buy_score += 10
                reasons.append("Mendekati level support")
            if pd.notna(row.get('resistance')) and row['Close'] >= row['resistance'] * 0.99:
                sell_score += 10
                reasons.append("Mendekati level resistance")
            
            max_score = 70
            buy_score = min(100, (buy_score / max_score) * 100)
            sell_score = min(100, (sell_score / max_score) * 100)
            return buy_score, sell_score, reasons
        
        # Hitung skor untuk 1D dan 4H
        buy_1d, sell_1d, reasons_1d = calculate_scores_from_row(row_1d)
        buy_4h, sell_4h, reasons_4h = calculate_scores_from_row(row_4h)
        
        # Gabungkan skor teknikal
        total_buy = buy_1d * tech_weights['1D'] + buy_4h * tech_weights['4h']
        total_sell = sell_1d * tech_weights['1D'] + sell_4h * tech_weights['4h']
        
        if total_buy > total_sell:
            tech_dir = "BUY"
            tech_conf = total_buy
        elif total_sell > total_buy:
            tech_dir = "SELL"
            tech_conf = total_sell
        else:
            tech_dir = "NEUTRAL"
            tech_conf = 50.0
        
        # Fundamental dianggap netral (bisa ditambahkan nanti)
        fund_dir = "NEUTRAL"
        fund_conf = 50.0
        
        # Gabungkan dengan bobot (sama seperti di analyzer)
        from analyzer import WEIGHT_TECHNICAL, WEIGHT_FUNDAMENTAL
        dir_val = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
        tech_val = dir_val[tech_dir] * tech_conf
        fund_val = dir_val[fund_dir] * fund_conf
        combined = WEIGHT_TECHNICAL * tech_val + WEIGHT_FUNDAMENTAL * fund_val
        if combined > 0:
            final_dir = "BUY"
            final_conf = abs(combined)
        elif combined < 0:
            final_dir = "SELL"
            final_conf = abs(combined)
        else:
            final_dir = "NEUTRAL"
            final_conf = 50.0
        
        # Jika final_conf >= 65 dan tidak dalam posisi, entry
        if final_conf >= 65 and not in_position:
            entry_price = row_4h['Close']
            atr = row_4h.get('volatility_atr', 0)
            if atr > 0:
                sl, tp = calculate_sl_tp(entry_price, atr, final_dir)
                in_position = True
                direction = final_dir
                entry_time = current_time
                # Catat trade dibuka
                trades.append({
                    'entry_time': entry_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'exit_time': None,
                    'exit_price': None,
                    'pnl': None,
                    'result': None
                })
        
        # Jika dalam posisi, cek apakah terkena SL atau TP
        elif in_position:
            # Cek apakah harga menyentuh SL atau TP
            if direction == "BUY":
                if row_4h['Low'] <= sl:
                    # Kena SL
                    exit_price = sl
                    pnl = (exit_price - entry_price) / entry_price
                    result = "LOSS"
                    # Tutup posisi
                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': result
                    })
                    in_position = False
                elif row_4h['High'] >= tp:
                    # Kena TP
                    exit_price = tp
                    pnl = (exit_price - entry_price) / entry_price
                    result = "WIN"
                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': result
                    })
                    in_position = False
            elif direction == "SELL":
                if row_4h['High'] >= sl:
                    exit_price = sl
                    pnl = (entry_price - exit_price) / entry_price  # untuk short
                    result = "LOSS"
                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': result
                    })
                    in_position = False
                elif row_4h['Low'] <= tp:
                    exit_price = tp
                    pnl = (entry_price - exit_price) / entry_price
                    result = "WIN"
                    trades[-1].update({
                        'exit_time': current_time,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'result': result
                    })
                    in_position = False
    
    # Jika masih ada posisi terbuka di akhir, tutup dengan harga close terakhir
    if in_position:
        last_row = df_4h.iloc[-1]
        exit_price = last_row['Close']
        if direction == "BUY":
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
        trades[-1].update({
            'exit_time': last_row.name,
            'exit_price': exit_price,
            'pnl': pnl,
            'result': 'OPEN'  # atau 'FORCE_CLOSE'
        })
    
    # Buat DataFrame trades
    df_trades = pd.DataFrame(trades)
    
    # Hitung statistik
    if len(df_trades) > 0:
        closed_trades = df_trades[df_trades['result'].isin(['WIN', 'LOSS'])]
        wins = closed_trades[closed_trades['result'] == 'WIN']
        losses = closed_trades[closed_trades['result'] == 'LOSS']
        win_rate = len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0
        total_pnl = closed_trades['pnl'].sum()
        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')
        
        print(f"=== Backtest untuk {pair} ===")
        print(f"Jumlah trade: {len(closed_trades)}")
        print(f"Win rate: {win_rate*100:.2f}%")
        print(f"Total PnL: {total_pnl*100:.2f}%")
        print(f"Profit factor: {profit_factor:.2f}")
        print(df_trades)
    else:
        print(f"Tidak ada trade yang memenuhi confidence >=65% untuk {pair}")

if __name__ == "__main__":
    # Contoh untuk EURUSD
    backtest_pair('EURUSD')
