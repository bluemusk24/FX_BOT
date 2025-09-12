import streamlit as st
import pandas as pd
import os

DATA_PATH = "local_data/tickers_df.parquet"

st.set_page_config(page_title="FX-BOT Predictions", layout="wide")
st.title("📈 FX-BOT Predictions Dashboard")

# Check if parquet file exists
if not os.path.exists(DATA_PATH):
    st.error(f"Predictions file not found: {DATA_PATH}. Run main.py first to generate it.")
    st.stop()

# Load predictions dataframe
df = pd.read_parquet(DATA_PATH)

# Sidebar selection
st.sidebar.header("🔧 Controls")

# Choose date
available_dates = sorted(df["Date"].dt.strftime("%Y-%m-%d").unique())
selected_date = st.sidebar.selectbox("Select Date:", available_dates, index=len(available_dates) - 1)

# Filter by date
date_df = df[df["Date"].dt.strftime("%Y-%m-%d") == selected_date].copy()

# Function to get signals
def get_signals(df, horizon, buy_thresh=0.60, sell_thresh=0.40, top_n=5):
    pred_col = f"pred_xgboost_{horizon}_best"
    prob_col = f"prob_pred_xgboost_{horizon}_best"

    if prob_col not in df.columns:
        prob_col = pred_col

    # Buy signals
    buy_signals = (
        df.loc[df[prob_col] >= buy_thresh, ["Ticker", prob_col]]
        .sort_values(by=prob_col, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # Sell signals
    sell_signals = (
        df.loc[df[prob_col] <= sell_thresh, ["Ticker", prob_col]]
        .sort_values(by=prob_col, ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    return buy_signals, sell_signals

# Display top buy/sell signals per horizon with colors
st.subheader(f"Top Signals on {selected_date}")

signals_dict = {}

for horizon in ["1h", "4h"]:
    st.markdown(f"### {horizon.upper()} Horizon")
    buy_signals, sell_signals = get_signals(date_df, horizon)
    signals_dict[horizon] = {"buy": buy_signals, "sell": sell_signals}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Buy Signals (Prob ≥ 0.60)")
        if buy_signals.empty:
            st.info("No strong buy signals.")
        else:
            st.dataframe(
                buy_signals.style.map(
                    lambda x: "color: green" if x in buy_signals["Ticker"].values else "",
                    subset=["Ticker"]
                ).background_gradient(
                    subset=[f"prob_pred_xgboost_{horizon}_best"], cmap="Greens"
                ),
                width="stretch"
            )

    with col2:
        st.markdown("#### Sell Signals (Prob ≤ 0.40)")
        if sell_signals.empty:
            st.info("No strong sell signals.")
        else:
            st.dataframe(
                sell_signals.style.map(
                    lambda x: "color: red" if x in sell_signals["Ticker"].values else "",
                    subset=["Ticker"]
                ).background_gradient(
                    subset=[f"prob_pred_xgboost_{horizon}_best"], cmap="Reds_r"
                ),
                width="stretch"
            )

# Strongest signals across both horizons (common tickers)
st.subheader("🔥 Strongest Signals Across Both Horizons")

# Strong buy tickers common to 1h AND 4h
common_strong_buy = set(signals_dict["1h"]["buy"]["Ticker"]).intersection(
    set(signals_dict["4h"]["buy"]["Ticker"])
)
# Strong sell tickers common to 1h AND 4h
common_strong_sell = set(signals_dict["1h"]["sell"]["Ticker"]).intersection(
    set(signals_dict["4h"]["sell"]["Ticker"])
)

# Display with colors
if common_strong_buy:
    st.markdown("#### Strong Buy Across Horizons")
    st.markdown(
        "<span style='color:green; font-weight:bold;'>"
        + ", ".join(sorted(common_strong_buy))
        + "</span>",
        unsafe_allow_html=True,
    )
else:
    st.info("No strong buy common to both horizons.")

if common_strong_sell:
    st.markdown("#### Strong Sell Across Horizons")
    st.markdown(
        "<span style='color:red; font-weight:bold;'>"
        + ", ".join(sorted(common_strong_sell))
        + "</span>",
        unsafe_allow_html=True,
    )
else:
    st.info("No strong sell common to both horizons.")
