import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.technicals import calculate_ema

# ==============================
# DASHBOARD UTAMA
# ==============================

st.set_page_config(page_title="Forex Dashboard", layout="wide")

st.title("📊 Forex Trading Dashboard")
st.markdown("Pantau pergerakan pasangan forex dengan indikator EMA-200 (4H).")

# --- Pilihan Pair ---
pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
selected_pair = st.sidebar.selectbox("Pilih Pasangan Forex:", pairs)

# --- Load Data ---
try:
    df = pd.read_csv(f"data/processed/{selected_pair}.csv")
    df['time'] = pd.to_datetime(df['time'])
except FileNotFoundError:
    st.error(f"Data untuk {selected_pair} belum tersedia. Jalankan scraper & preprocessing terlebih dahulu.")
    st.stop()

# --- Hitung EMA 200 (timeframe 4H) ---
df = calculate_ema(df, column="close", period=200)

# --- Chart Harga ---
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name="Harga"
))

# EMA-200
fig.add_trace(go.Scatter(
    x=df['time'],
    y=df['EMA_200'],
    mode="lines",
    line=dict(color="orange", width=2),
    name="EMA 200"
))

fig.update_layout(
    title=f"{selected_pair} dengan EMA 200 (4H)",
    xaxis_title="Waktu",
    yaxis_title="Harga",
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# --- Info Fundamental ---
st.subheader("📰 Berita Fundamental Terkini")
try:
    news_df = pd.read_csv("data/processed/news.csv").tail(10)
    st.table(news_df[['time', 'title', 'impact']])
except FileNotFoundError:
    st.warning("Data berita fundamental belum tersedia. Jalankan `scraper_fundamental.py` dulu.")
