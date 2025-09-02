import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="FX-BOT Results", layout="wide")
st.title("FX-BOT: Currency Pair Analysis Results")

# Path to local data directory
data_dir = "local_data/"

# Show available data files
data_files = os.listdir(data_dir) if os.path.exists(data_dir) else []

if not data_files:
    st.warning("No results found. Please run the data pipeline first.")
else:
    st.sidebar.header("Available Data Files")
    for file in data_files:
        st.sidebar.write(file)

    # Show model predictions and best currency pair to invest in
    tickers_path = os.path.join(data_dir, "tickers_df.parquet")
    if os.path.exists(tickers_path):
        df = pd.read_parquet(tickers_path)
        st.subheader("Currency Pairs: Model Predictions and Ranks")
        # Always try to use XGBoost prediction columns if available
        pred_col = None
        prob_col = None
        rank_col = None
        if 'pred_xgboost_1h_best' in df.columns:
            pred_col = 'pred_xgboost_1h_best'
        if 'prob_xgboost_1h_best' in df.columns:
            prob_col = 'prob_xgboost_1h_best'
        if 'pred_xgboost_1h_best_rank' in df.columns:
            rank_col = 'pred_xgboost_1h_best_rank'
        # Fallback to auto-detect if not found
        if not pred_col or not rank_col:
            for col in df.columns:
                if not rank_col and col.endswith('_rank'):
                    rank_col = col
                if not pred_col and (col.startswith('pred') or col.startswith('y_pred') or col.startswith('class')):
                    pred_col = col
        # If still not found, fallback to default names
        if not rank_col:
            rank_col = 'pred_class1_rank' if 'pred_class1_rank' in df.columns else df.columns[-1]
        if not pred_col:
            pred_col = 'pred_class1' if 'pred_class1' in df.columns else df.columns[-2]

        # Filter for each date and show top/bottom 3 predictions per day
        if 'Date' in df.columns:
            st.write(f"### Forex Currency Pairs Daily Predictions (1-hour future growth)")
            def highlight_buy(val):
                if isinstance(val, (int, float)) and val >= 0.5:
                    return 'background-color: #d4f7d4'  # light green for buy
                elif isinstance(val, (int, float)):
                    return 'background-color: #f7d4d4'  # light red for sell
                return ''

            for date in sorted(df['Date'].unique()):
                df_day = df[df['Date'] == date].copy()
                # Format date for section header to YYYY-MM-DD only
                date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                st.markdown(f"#### {date_str}")
                # Find probability column
                prob_col = None
                for col in df_day.columns:
                    if 'prob' in col or 'proba' in col:
                        prob_col = col
                        break
                # Filter for XGBoost Buy predictions only
                if pred_col in df_day.columns and 'is_positive_growth_1h_future' in df_day.columns:
                    # Filter for both positive actual growth and XGBoost probability >= 0.56
                    if prob_col and prob_col in df_day.columns:
                        growth = df_day[(df_day['is_positive_growth_1h_future'] == 1) & (df_day[prob_col] >= 0.56)].copy()
                    else:
                        growth = df_day[(df_day['is_positive_growth_1h_future'] == 1) & (df_day[pred_col] == 1)].copy()
                    if not growth.empty:
                        # Sort by probability if available
                        if prob_col and prob_col in growth.columns:
                            growth = growth.sort_values(prob_col, ascending=False)
                            display_cols = ["Ticker", prob_col, pred_col]
                            growth = growth[display_cols].rename(columns={prob_col: 'Probability', pred_col: 'Growth_Signal'})
                        else:
                            display_cols = ["Ticker", pred_col]
                            growth = growth[display_cols].rename(columns={pred_col: 'Growth_Signal'})
                        growth = growth.reset_index(drop=True)
                        st.dataframe(growth)
                        st.caption('Showing only pairs with actual positive growth and model probability >= 0.56')
                    else:
                        st.info('No qualifying predictions for this day.')
        else:
            st.warning("No 'Date' column found in tickers_df.parquet.")
    else:
        st.info("tickers_df.parquet not found.")


st.markdown("---")
st.caption("FX-BOT Streamlit UI | Results Viewer")
