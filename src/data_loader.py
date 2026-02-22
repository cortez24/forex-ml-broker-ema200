import os
import pandas as pd
import requests
from datetime import datetime, timedelta
from config import DATA_FOLDER, TWELVEDATA_API_KEY

def load_historical_data(pair, timeframe):
    """
    Memuat data historis dari file CSV.
    Format file: datetime, open, high, low, close, volume (tanpa header, delimiter tab)
    Nama file: {pair}_{timeframe}.csv disimpan di folder DATA_FOLDER.
    """
    filename = f"{pair}_{timeframe}.csv"
    filepath = os.path.join(DATA_FOLDER, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File data tidak ditemukan: {filepath}")
    
    # Baca CSV tanpa header, nama kolom diberikan manual
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        parse_dates=['datetime']
    )
    
    # Urutkan berdasarkan datetime (jika belum urut)
    df.sort_values('datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def get_realtime_price(pair):
    """
    Mendapatkan harga real-time dari Twelve Data API.
    Mengembalikan dict dengan harga terbaru (open, high, low, close, volume) atau None jika gagal.
    """
    url = f"https://api.twelvedata.com/quote"
    params = {
        "symbol": pair,
        "apikey": TWELVEDATA_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if response.status_code == 200 and 'close' in data:
            # Twelve Data mengembalikan banyak field, kita ambil yang diperlukan
            return {
                'open': float(data['open']),
                'high': float(data['high']),
                'low': float(data['low']),
                'close': float(data['close']),
                'volume': float(data['volume']) if data['volume'] else 0,
                'datetime': datetime.now()
            }
        else:
            print("Error from Twelve Data:", data.get('message', 'Unknown error'))
            return None
    except Exception as e:
        print("Exception saat mengambil data real-time:", e)
        return None

def get_historical_last_candle(pair, timeframe):
    """
    Mengambil candle terakhir dari data historis.
    Digunakan untuk sinyal (kombinasi dengan real-time jika perlu).
    """
    df = load_historical_data(pair, timeframe)
    if df.empty:
        return None
    last = df.iloc[-1].to_dict()
    # pastikan datetime dalam format string jika diperlukan
    last['datetime'] = last['datetime'].isoformat()
    return last
