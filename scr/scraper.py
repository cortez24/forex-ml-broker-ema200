# src/scraper.py
import os
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List
import yfinance as yf

from config import SUPPORTED_PAIRS, RAW_DIR, DEFAULT_PERIOD, DEFAULT_INTERVAL

os.makedirs(RAW_DIR, exist_ok=True)

# -------------------------
# Price fetcher (yfinance fallback)
# -------------------------
def fetch_price_yf(pair: str, period: str = DEFAULT_PERIOD, interval: str = DEFAULT_INTERVAL):
    """
    pair: 'EUR/USD' -> map to 'EURUSD=X'
    interval: '4h' -> yfinance expects '240m' or '4h' accepted
    """
    base, quote = pair.split('/')
    ticker = f"{base}{quote}=X"
    # yfinance interval expects '240m' or '1d', '1h' etc. We'll use '240m'
    if interval.lower().endswith('h'):
        minutes = int(interval[:-1]) * 60
        yf_interval = f"{minutes}m"
    else:
        yf_interval = interval
    df = yf.download(ticker, period=period, interval=yf_interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={'Datetime':'time', 'Date':'time'})
    # unify column names
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df['time'] = pd.to_datetime(df['time'])
    return df[['time','open','high','low','close','volume']]

# -------------------------
# News scraper (Investing.com)
# -------------------------
def scrape_investing_news(pages: int = 1, limit=20) -> pd.DataFrame:
    base = "https://www.investing.com/news/forex-news"
    headers = {"User-Agent":"Mozilla/5.0"}
    articles = []
    try:
        r = requests.get(base, headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print("News fetch failed:", e)
        return pd.DataFrame()
    soup = BeautifulSoup(r.text, "lxml")
    nodes = soup.select(".textDiv")[:limit]
    for node in nodes:
        title = node.get_text(strip=True)
        link_tag = node.find_parent("a", href=True)
        link = "https://www.investing.com" + link_tag['href'] if link_tag and link_tag['href'].startswith('/') else (link_tag['href'] if link_tag else "")
        articles.append({'time': datetime.utcnow(), 'title': title, 'url': link})
    if not articles:
        return pd.DataFrame()
    df = pd.DataFrame(articles)
    df['time'] = pd.to_datetime(df['time'])
    return df

# -------------------------
# Runner: fetch all pairs & news
# -------------------------
def run_all(fetch_pairs: List[str] = None):
    if fetch_pairs is None:
        fetch_pairs = SUPPORTED_PAIRS
    for pair in fetch_pairs:
        try:
            df = fetch_price_yf(pair)
            if df.empty:
                print(f"[WARN] empty data for {pair}")
                continue
            symbol = pair.replace('/', '')
            path = os.path.join(RAW_DIR, f"{symbol}.csv")
            df.to_csv(path, index=False)
            print(f"[OK] saved {path} rows={len(df)}")
            time.sleep(1)
        except Exception as e:
            print(f"[ERROR] {pair}: {e}")

    news = scrape_investing_news()
    if not news.empty:
        npath = os.path.join(RAW_DIR, "news.csv")
        news.to_csv(npath, index=False)
        print(f"[OK] saved news to {npath}")

if __name__ == "__main__":
    run_all()
