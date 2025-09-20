# src/dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
from config import PROCESSED_DIR, SUPPORTED_PAIRS
from predict import predict_next

st.set_page_config(page_title="Forex ML Broker (Full)", layout="wide")

st.title("Forex ML Broker — EMA200 (4H) + MACD + XGBoost")

pairs = SUPPORTED_PAIRS
pair = st.sidebar.selectbox("Pair", pairs)
symbol = pair.replace('/','')

# load processed data for plotting
path = os.path.join(PROCESSED_DIR, f"{symbol}_features.csv")
if not os.path.exists(path):
    st.warning(f"Data {path} tidak ditemukan. Jalankan scraper & feature preparation.")
    st.stop()
df = pd.read_csv(path, parse_dates=['time'])
# plot last 200 bars
plot_df = df.tail(200)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=plot_df['time'],
    open=plot_df['open'],
    high=plot_df['high'],
    low=plot_df['low'],
    close=plot_df['close'],
    name='Price'
))
fig.add_trace(go.Scatter(x=plot_df['time'], y=plot_df['EMA200'], name='EMA200', line=dict(color='orange', width=1.5)))
fig.update_layout(title=f"{pair} (last 200)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("ML Prediction & Trade Suggestion")
try:
    res = predict_next(symbol)
    st.metric("Prediction", f"{res['prediction']} (P up: {res['prob_up']*100:.2f}%)")
    st.write(f"Last price: {res['last_price']}")
    if res['tp_price'] is not None:
        st.write(f"Take Profit: {res['tp_price']:.5f}  Stop Loss: {res['sl_price']:.5f}")
    else:
        st.info("No clear signal (probability near neutral).")
except Exception as e:
    st.error(str(e))
