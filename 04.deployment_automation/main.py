from scripts.data_repo import DataRepository
from scripts.transform import TransformData
from scripts.train import TrainModel

import pandas as pd
import warnings
import os
from datetime import datetime  

def main():
    FETCH_REPO = True
    TRANSFORM_DATA = True
    TRAIN_MODEL = True

    data_dir = 'local_data/'
    ticker_file = os.path.join(data_dir, 'tickers_df.parquet')

    # Step 1: Get data
    print('Step 1: Getting data from APIs or Load from disk')
    repo = DataRepository()
    if FETCH_REPO:
        try:
            repo.fetch()
            repo.persist(data_dir=data_dir)
        except Exception as e:
            print(f"Failed to fetch fresh data: {e}")
            if os.path.exists(ticker_file):
                repo.load(data_dir=data_dir)
            else:
                return
    else:
        if os.path.exists(ticker_file):
            repo.load(data_dir=data_dir)
        else:
            return

    # Step 2: Transform data
    print('Step 2: Transforming data into one dataframe')
    transformed = TransformData(repo=repo)
    if TRANSFORM_DATA:
        transformed.transform()
        transformed.persist(data_dir=data_dir)
    else:
        transformed.load(data_dir=data_dir)

    # Step 3: Train/Load Model
    print('Step 3: Training the model or loading from disk')
    warnings.filterwarnings("ignore")
    trained = TrainModel(transformed=transformed)

    if TRAIN_MODEL:
        trained.prepare_dataframe()
        trained.train_xgboost()  # trains both 1h and 4h
        trained.persist(data_dir=data_dir)
    else:
        trained.prepare_dataframe()
        trained.load(data_dir=data_dir)

    # Step 4: Make Inference
    print('Step 4: Making inference')
    trained.make_inference()  

    # Show only buy signals (prediction >= 0.56)
    for horizon in ["1h", "4h"]:
        pred_col = f"pred_xgboost_{horizon}_best"
        prob_col = f"prob_pred_xgboost_{horizon}_best"

        print(f"\n=== {horizon.upper()} Buy Predictions (prob >= 0.6) ===")
        buy_signals = trained.df_full.loc[
            trained.df_full[pred_col] & (trained.df_full[prob_col] >= 0.60),
            ["Ticker", pred_col, prob_col]  
        ]

        if buy_signals.empty:
            print("No buy signals for this horizon.")
        else:
            print(buy_signals.sort_values(by=prob_col, ascending=False).reset_index(drop=True))

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\nCurrent date and time: {current_datetime}")


if __name__ == "__main__":
    main()
    print("\nLaunching Streamlit UI...")
    os.system("streamlit run streamlit_app.py")
