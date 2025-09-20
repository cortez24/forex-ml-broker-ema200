import pandas as pd

# ==============================
# TEKNIKAL INDICATORS
# ==============================

def add_ema(df: pd.DataFrame, period: int = 200) -> pd.DataFrame:
    """Tambah EMA ke dataframe."""
    df[f"EMA_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame, short=12, long=26, signal=9) -> pd.DataFrame:
    """Tambah indikator MACD (garis MACD & signal)."""
    df["EMA_short"] = df["close"].ewm(span=short, adjust=False).mean()
    df["EMA_long"] = df["close"].ewm(span=long, adjust=False).mean()
    df["MACD"] = df["EMA_short"] - df["EMA_long"]
    df["Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Tambah RSI ke dataframe."""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


# ==============================
# FUNDAMENTAL (NEWS)
# ==============================

def merge_fundamental(price_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gabungkan data harga dengan berita fundamental.
    Misalnya: one-hot impact level.
    """
    news_df = news_df.copy()
    news_df["time"] = pd.to_datetime(news_df["time"])
    news_df = pd.get_dummies(news_df, columns=["impact"])

    # Merge on nearest timestamp (forward fill)
    merged = pd.merge_asof(
        price_df.sort_values("time"),
        news_df.sort_values("time"),
        on="time",
        direction="backward"
    )
    return merged.fillna(0)


# ==============================
# MASTER FUNCTION
# ==============================

def prepare_features(price_df: pd.DataFrame, news_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Siapkan semua fitur teknikal + fundamental.
    """
    df = price_df.copy()
    df = add_ema(df, period=200)
    df = add_macd(df)
    df = add_rsi(df)

    if news_df is not None:
        df = merge_fundamental(df, news_df)

    # Drop rows dengan NaN akibat rolling
    df = df.dropna().reset_index(drop=True)
    return df

