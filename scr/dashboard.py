import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

from predict import predict_next_move


# ==============================
# LOAD DATA
# ==============================
def load_data(pair, data_dir="data/processed"):
    file_path = os.path.join(data_dir, f"{pair}.csv")
    if not os.path.exists(file_path):
        st.error(f"Data {pair} tidak ditemukan di {file_path}")
        return None
    df = pd.read_csv(file_path)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ==============================
# DASHBOARD APP
# ==============================
def main():
    st.set_page_config(page_title="Forex Trading Dashboard", layout="wide")

    st.title("📊 Forex Trading Dashboard dengan ML Predictions")

    pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
    pair = st.sidebar.selectbox("Pilih Pair", pairs)

    df = load_data(pair)
    if df is None:
        return

    # Chart candlestick
    fig = go.Figure(data=[go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=pair
    )])

    fig.update_layout(
        title=f"{pair} - Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # ML PREDICTION
    # ==============================
    st.subheader("🤖 Prediksi Machine Learning")

    try:
        result = predict_next_move(pair)
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Pair", result["pair"])
            st.metric("Last Price", f"{result['last_price']:.4f}")
            st.metric("Prediction", result["prediction"])

        with col2:
            st.metric("Prob UP", f"{result['probability_up']*100:.2f}%")
            st.metric("Prob DOWN", f"{result['probability_down']*100:.2f}%")
            st.metric("Timestamp", result["timestamp"])

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")


if __name__ == "__main__":
    main()
