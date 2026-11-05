"""Streamlit dashboard. Run locally:  streamlit run app.py"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_prices, add_features
from src.models import time_split, naive_forecast, arima_forecast, evaluate, walk_forward

st.set_page_config(page_title="Crypto Time Series", layout="wide")
st.title("Cryptocurrency Time Series Dashboard")
st.caption("Explore price, returns and volatility, then compare ARIMA with a naive baseline.")


def show(fig):
    """Display a plotly figure; works on old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


# ---------------- sidebar ----------------
coin = st.sidebar.selectbox("Coin", ["BTC-USD", "ETH-USD", "SOL-USD"])
years = st.sidebar.slider("Years of history", 2, 5, 4)
window = st.sidebar.slider("Rolling window (days)", 7, 90, 30)
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 60, 14)
p = st.sidebar.number_input("ARIMA p", 0, 5, 1)
q = st.sidebar.number_input("ARIMA q", 0, 5, 1)


@st.cache_data(ttl=3600, show_spinner="Loading data...")
def get_data(ticker, years):
    return load_prices(ticker, years=years, refresh=True)


try:
    raw = get_data(coin, years)
except Exception:
    st.error("Could not download data right now. Please try again in a minute.")
    st.stop()

df = add_features(raw, window)
close = df["Close"]

tab1, tab2, tab3 = st.tabs(["Price", "Returns & volatility", "Forecast"])

# ---------------- tab 1 ----------------
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest price", f"${close.iloc[-1]:,.0f}")
    c2.metric("30-day change", f"{(close.iloc[-1] / close.iloc[-31] - 1) * 100:.1f}%")
    c3.metric("Annualised volatility", f"{df['Volatility'].iloc[-1] * 100:.0f}%")
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=close, name="Close")
    fig.add_scatter(x=df.index, y=df["MA"], name=f"{window}-day MA")
    fig.update_layout(title=f"{coin} price", yaxis_type="log" if st.checkbox("Log scale") else "linear")
    show(fig)

# ---------------- tab 2 ----------------
with tab2:
    ret = df["Return"].dropna()
    fig = go.Figure(go.Histogram(x=ret, nbinsx=80))
    fig.update_layout(title="Distribution of daily log returns (note the fat tails)")
    show(fig)
    st.line_chart(df["Volatility"])
    st.write(f"Skewness: **{ret.skew():.2f}**, excess kurtosis: **{ret.kurt():.2f}** (0 would be a normal distribution).")

# ---------------- tab 3 ----------------
with tab3:
    st.write("The last *horizon* days are hidden from the model and then predicted.")
    train, test = time_split(close, horizon)
    order = (int(p), 1, int(q))
    preds = {"Naive": naive_forecast(train, horizon), f"ARIMA{order}": arima_forecast(train, horizon, order)}

    table = pd.DataFrame({k: evaluate(test.values, v) for k, v in preds.items()}).T
    st.dataframe(table.round(2))

    fig = go.Figure()
    fig.add_scatter(x=train.index[-90:], y=train.iloc[-90:], name="Train", line=dict(color="gray"))
    fig.add_scatter(x=test.index, y=test, name="Actual", line=dict(color="black", width=3))
    for name, pred in preds.items():
        fig.add_scatter(x=test.index, y=pred, name=name, line=dict(dash="dash"))
    show(fig)

    @st.cache_data(show_spinner="Running walk-forward validation...")
    def run_walk_forward(series, horizon, order):
        fns = {"Naive": naive_forecast, f"ARIMA{order}": lambda tr, h: arima_forecast(tr, h, order)}
        return walk_forward(series, fns, horizon=horizon, n_folds=8)

    st.subheader("Walk-forward validation (8 folds)")
    try:
        st.dataframe(run_walk_forward(close, horizon, order))
        st.caption("Beat_Naive_% = share of folds where the model had lower RMSE than the naive baseline.")
    except ValueError as e:
        st.info(str(e))

st.caption("Educational project. Not financial advice.")
