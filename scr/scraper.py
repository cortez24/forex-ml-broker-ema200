# src/scraper.py
"""
Scraper gabungan untuk:
1. Data harga forex (multi-pair dari Alpha Vantage API)
2. Data berita fundamental (dari Investing.com)

Output:
- data/raw/forex_<pair>.csv   (misalnya forex_usd_idr.csv)
- data/raw/news_data.csv
"""

import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# === Konfigurasi API ===
API_KEY = "demo"  # Ganti dengan API key Alpha Vantage Anda
BASE_URL = "https://www.alphavantage.co/query"

# === Konfigurasi path ===
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# === Pasangan forex yang akan diambil ===
PAIRS = ["USD/IDR", "EUR/USD", "AUD/USD"]
INTERVAL = "5min"  # bisa "5min", "15min", "60min"

# === Scraper Forex ===
def fetch_forex(pair, interval="5min", outputsize="compact"):
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": pair.split("/")[0],
        "to_symbol": pair.split("/")[1],
        "interval": interval,
        "apikey": API_KEY
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()

    key = f"Time Series FX ({interval})"
    if key not in data:
        print(f"❌ Gagal ambil data forex untuk {pair}.")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(data[key], orient="index")
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close"
    }).astype(float)

    df = df.sort_index().reset_index().rename(columns={"index": "timestamp"})
    df["pair"] = pair
    return df

# === Scraper News ===
def fetch_news(url="https://www.investing.com/news/economy"):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        print("❌ Gagal ambil berita.")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "html.parser")
    articles = soup.find_all("article", limit=10)

    news_data = []
    for art in articles:
        title = art.get_text().strip()
        link = art.find("a")["href"] if art.find("a") else ""
        news_data.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "title": title,
            "link": "https://www.investing.com" + link if link else ""
        })

    return pd.DataFrame(news_data)

# === Main ===
if __name__ == "__main__":
    # Scrape forex untuk semua pair
    for pair in PAIRS:
        forex_df = fetch_forex(pair, INTERVAL)
        if not forex_df.empty:
            pair_name = pair.replace("/", "_").lower()
            forex_path = os.path.join(RAW_DIR, f"forex_{pair_name}.csv")
            forex_df.to_csv(forex_path, index=False)
            print(f"✅ Data forex {pair} disimpan di {forex_path}")
        time.sleep(12)  # rate limit per request

    # Scrape news
    news_df = fetch_news()
    if not news_df.empty:
        news_path = os.path.join(RAW_DIR, "news_data.csv")
        news_df.to_csv(news_path, index=False)
        print(f"✅ Data berita disimpan di {news_path}")
