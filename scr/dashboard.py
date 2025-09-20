# src/dashboard.py
"""
Dashboard Forex ML Broker
Menampilkan grafik harga forex + indikator teknikal (EMA200, MACD).
Menggunakan Streamlit.
"""

import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.technicals import add_indicators

# === Konfigurasi ===
DATA_DIR = "data/processed"
BACKTEST_DIR = "data/backtest"

# === Fungsi Load Data ===
def load_data(pair="usd_idr"):
    file_path = os.path.join(DATA_DIR, f"forex_{pair}_with_indicators.csv")
    if not os.path.exists(file_path):
        st.error(f"Data untuk {pair.upper()} belum tersedia. Jalankan 01_data_preparation.ipynb dulu.")
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    return add_indicators(df)

# === Fungsi Chart Harga + Indikator ===
def plot_chart(df, pair="USD/IDR"):
    fig = go.Figure()

    # Harga close
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["close"],
        mode="lines",
        name="Close Price"
    ))

    # EMA200
    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["ema200"],
        mode="lines",
        name="EMA200"
    ))

    # Tambahkan sinyal buy/sell
    buy_signals = df[df["signal"] == 1]
    sell_signals = df[df["signal"] == -1]

    fig.add_trace(go.Scatter(
        x=buy_signals["timestamp"],
        y=buy_signals["close"],
        mode="markers",
        marker=dict(color="green", size=10, symbol="triangle-up"),
        name="Buy Signal"
    ))

    fig.add_trace(go.Scatter(
        x=sell_signals["timestamp"],
        y=sell_signals["close"],
        mode="markers",
        marker=dict(color="red", size=10, symbol="triangle-down"),
        name="Sell Signal"
    ))

    fig.update_layout(
        title=f"Forex {pair} dengan EMA200 & Sinyal Trading",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white"
    )
    return fig

# === Fungsi Plot MACD ===
def plot_macd(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["macd_line"],
        mode="lines",
        name="MACD Line"
    ))

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["signal_line"],
        mode="lines",
        name="Signal Line"
    ))

    fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["macd_histogram"],
        name="Histogram"
    ))

    fig.update_layout(
        title="MACD Indicator",
        xaxis_title="Time",
        yaxis_title="MACD",
        template="plotly_white"
    )
    return fig

# === Fungsi Ringkasan Backtest ===
def load_backtest(pair="usd_idr"):
    file_path = os.path.join(BACKTEST_DIR, f"backtest_{pair}.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    total_return = df["cumulative_strategy"].iloc[-1] - 1
    win_rate = (df["strategy_return"] > 0).sum() / len(df)
    return {
        "total_return": total_return,
        "win_rate": win_rate,
        "trades": (df["signal"] != 0).sum()
    }

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="Forex ML Broker Dashboard", layout="wide")
    st.title("📊 Forex ML Broker Dashboard")

    # Pilihan Pair
    pairs = ["usd_idr", "eur_usd", "aud_usd"]
    pair = st.sidebar.selectbox("Pilih Pair", pairs)

    # Load data
    df = load_data(pair)
    if df.empty:
        return

    # Tambahkan sinyal
    df["signal"] = 0
    df.loc[(df["close"] > df["ema200"]) & (df["macd_line"] > df["signal_line"]), "signal"] = 1
    df.loc[(df["close"] < df["ema200"]) & (df["macd_line"] < df["signal_line"]), "signal"] = -1

    # Chart harga + EMA200
    st.plotly_chart(plot_chart(df, pair.upper()), use_container_width=True)

    # Chart MACD
    st.plotly_chart(plot_macd(df), use_container_width=True)

    # Ringkasan backtest
    st.subheader("📈 Backtest Summary")
    stats = load_backtest(pair)
    if stats:
        st.metric("Total Return", f"{stats['total_return']:.2%}")
        st.metric("Win Rate", f"{stats['win_rate']:.2%}")
        st.metric("Total Trades", stats["trades"])
    else:
        st.warning("⚠️ Belum ada hasil backtest untuk pair ini. Jalankan 02_backtest.ipynb dulu.")

if __name__ == "__main__":
    main()

