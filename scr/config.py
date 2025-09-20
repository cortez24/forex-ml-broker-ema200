# src/config.py
SUPPORTED_PAIRS = [
    "EUR/USD",
    "USD/IDR",
    "AUD/USD",
    "GBP/JPY",
    "EUR/JPY",
    "CHF/JPY",
    "USD/JPY"
]

# Default directories
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models"
BACKTEST_DIR = "data/backtest"

# Scraper / data settings
DEFAULT_PERIOD = "6mo"
DEFAULT_INTERVAL = "4h"  # used when fetching via yfinance
