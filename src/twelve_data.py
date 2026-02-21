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

env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)
API_KEY = os.getenv("TWELVE_DATA_API_KEY")
BASE_URL = "https://api.twelvedata.com"

REQUEST_DELAY = 8
last_request_time = 0

def rate_limit():
    global last_request_time
    current_time = time.time()
    time_since_last = current_time - last_request_time
    if time_since_last < REQUEST_DELAY:
        sleep_time = REQUEST_DELAY - time_since_last
        time.sleep(sleep_time)
    last_request_time = time.time()

def get_real_time_price(symbol):
    rate_limit()
    url = f"{BASE_URL}/price"
    params = {"symbol": symbol, "apikey": API_KEY, "dp": 5}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "price" in data:
            return float(data["price"])
        else:
            logging.error(f"Price API error: {data.get('message', 'Unknown')}")
            return None
    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {e}")
        return None

def get_quote(symbol):
    rate_limit()
    url = f"{BASE_URL}/quote"
    params = {"symbol": symbol, "apikey": API_KEY, "dp": 5}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "price" in data:
            return data
        else:
            logging.error(f"Quote error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"Error fetching quote for {symbol}: {e}")
        return None

def get_historical_data(symbol, interval="1h", outputsize=5000, start_date=None, end_date=None):
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
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "values" in data:
            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.set_index("datetime", inplace=True)
            # Ubah nama kolom ke format yang kita pakai (Capitalized)
            rename_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }
            # Hanya rename kolom yang ada
            for old, new in rename_map.items():
                if old in df.columns:
                    df.rename(columns={old: new}, inplace=True)
            
            # Pastikan kolom Open, High, Low, Close ada
            required = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required):
                logging.error(f"Data dari API tidak memiliki kolom OHLC. Kolom: {df.columns.tolist()}")
                return None
            
            # Jika Volume tidak ada, tambahkan dengan nilai 0
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            
            # Konversi ke numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)
            return df
        elif "code" in data:
            logging.error(f"API Error: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        logging.error(f"Error fetching historical data for {symbol}: {e}")
        return None

def download_and_cache(symbol, interval, output_folder="data/api"):
    base_path = Path(output_folder)
    base_path.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-")
    filename = f"{safe_symbol}_{interval}.csv"
    filepath = base_path / filename

    if filepath.exists():
        file_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        if datetime.now() - file_time < timedelta(days=1):
            logging.info(f"Menggunakan cached data: {filename}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            # Pastikan kolom berformat Title Case (Open, High, dll)
            df.columns = [col.capitalize() for col in df.columns]
            # Jika Volume tidak ada, tambahkan
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            return df

    logging.info(f"Downloading {symbol} {interval}...")
    df = get_historical_data(symbol, interval)
    if df is not None and not df.empty:
        df.to_csv(filepath)
        logging.info(f"Saved to {filepath}")
        return df
    else:
        return None

def get_intraday_data(symbol, interval="1h", days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    if interval in ["1min", "5min"] and days > 7:
        chunk_days = 7
        chunks = []
        current_start = start_date
        while current_start < end_date:
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            df_chunk = get_historical_data(
                symbol, interval,
                start_date=current_start.strftime("%Y-%m-%d"),
                end_date=chunk_end.strftime("%Y-%m-%d")
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
