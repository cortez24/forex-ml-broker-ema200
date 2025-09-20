import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_forex_news():
    url = "https://www.investing.com/news/forex-news"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    articles = []
    for item in soup.select(".articleItem"):
        title = item.select_one(".title").get_text(strip=True)
        link = "https://www.investing.com" + item.select_one("a")["href"]
        articles.append({"title": title, "link": link})
    return pd.DataFrame(articles)

if __name__ == "__main__":
    df = scrape_forex_news()
    df.to_csv("../data/news/forex_news.csv", index=False)
    print("✅ News saved!")

