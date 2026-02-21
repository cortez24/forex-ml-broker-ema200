# twelve_data.py
import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com"

# Rate limiting untuk free tier (8 detik antara request)
REQUEST_DELAY = 8
last_request_time = 0

def rate_limit():
    """Menerapkan rate limiting untuk free tier"""
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < REQUEST_DELAY:
        sleep_time = REQUEST_DELAY - time_since_last
        time.sleep(sleep_time)
    last_request_time = time.time()

def get_real_time_price(symbol):
    """
    Mendapatkan harga real-time sederhana dari Twelve Data
    """
    rate_limit()
    url = f"{BASE_URL}/price"
    params = {
        "symbol": symbol,
        "apikey": API_KEY,
        "dp": 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "price" in data:
            return float(data["price"])
        elif "code" in data:
            logging.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {e}")
        return None

def get_quote(symbol):
    """
    Mendapatkan quote lengkap (bid, ask, volume, dll)
    """
    rate_limit()
    url = f"{BASE_URL}/quote"
    params = {
        "symbol": symbol,
        "apikey": API_KEY,
        "dp": 5
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if "price" in data:
            return data
        else:
            logging.error(f"Quote error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"Error fetching quote for {symbol}: {e}")
        return None

def get_historical_data(symbol, interval="1h", outputsize=5000, start_date=None, end_date=None):
    """
    Mendapatkan data historis OHLCV
    """
    rate_limit()
    url = f"{BASE_URL}/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": outputsize,
        "timezone": "UTC",
        "order": "ASC"
    }
    
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if "values" in data:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            df.columns = [col.capitalize() for col in df.columns]
            return df
        elif "code" in data:
            logging.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def download_and_cache(symbol, interval, output_folder="data/api"):
    """
    Mendownload data dan menyimpannya ke file CSV lokal untuk caching
    """
    base_path = Path(output_folder)
    base_path.mkdir(parents=True, exist_ok=True)
    
    safe_symbol = symbol.replace("/", "-")
    filename = f"{safe_symbol}_{interval}.csv"
    filepath = base_path / filename
    
    # Cek cache (kurang dari 1 hari)
    if filepath.exists():
        file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        if datetime.now() - file_time < timedelta(days=1):
            logging.info(f"Menggunakan cached data: {filename}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return df
    
    # Download data baru
    logging.info(f"Downloading {symbol} {interval}...")
    df = get_historical_data(symbol, interval)
    
    if df is not None and not df.empty:
        df.to_csv(filepath)
        logging.info(f"Saved to {filepath}")
        return df
    else:
        return None

def get_intraday_data(symbol, interval="1h", days=30):
    """
    Mendapatkan data intraday untuk periode tertentu (dengan pagination jika perlu)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Untuk interval kecil, perlu multiple chunks
    if interval in ["1min", "5min"] and days > 7:
        chunk_days = 7
        chunks = []
        current_start = start_date
        
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            start_chunk = current_start.strftime("%Y-%m-%d")
            end_chunk = chunk_end.strftime("%Y-%m-%d")
            
            df_chunk = get_historical_data(
                symbol, 
                interval, 
                start_date=start_chunk, 
                end_date=end_chunk
            )
            
            if df_chunk is not None and not df_chunk.empty:
                chunks.append(df_chunk)
            
            current_start = chunk_end
        
        if chunks:
            return pd.concat(chunks).sort_index()
        else:
            return None
    else:
        return get_historical_data(symbol, interval, start_date=start_str, end_date=end_str)
