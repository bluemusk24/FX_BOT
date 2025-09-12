import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb

xgb.set_config(verbosity=0)


class TrainModel:
    def __init__(self, transformed):
        # Correct dataframe from TransformData
        self.df_full = transformed.transformed_df.copy()
        self.df_full['ln_volume'] = self.df_full.Volume.apply(lambda x: np.log(x) if x > 0 else np.nan)

        self.models = {}  # will hold models for 1h and 4h
        self.prediction_cols = []

    # --------------------------
    # Feature sets
    # --------------------------
    def _define_feature_sets(self):
        self.GROWTH = [g for g in self.df_full if (g.find('growth_') == 0) & (g.find('future') < 0)]
        self.OHLCV = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.CATEGORICAL = ['Month', 'Weekday', 'Ticker']
        self.TO_PREDICT = [g for g in self.df_full.keys() if (g.find('future') >= 0)]
        self.MACRO = [
            'gdppot_us_yoy', 'gdppot_us_qoq', 'cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS',
            'trade_balance_us_yoy', 'trade_balance_us_mom', 'DGS1', 'DGS5', 'DGS10'
        ]
        self.CUSTOM_NUMERICAL = [
            'SMA10', 'SMA20', 'growing_moving_average',
            'high_minus_low_relative', 'volatility', 'ln_volume'
        ]

        self.TO_DROP = ['Year', 'Date', 'index_x', 'index_y', 'index', 'Quarter'] + self.CATEGORICAL + self.OHLCV

        self.TECHNICAL_INDICATORS = [
            'adx', 'adxr', 'apo', 'aroon_1', 'aroon_2', 'aroonosc', 'bop', 'cci', 'cmo',
            'dx', 'macd', 'macdsignal', 'macdhist', 'macd_ext', 'macdsignal_ext', 'macdhist_ext',
            'macd_fix', 'macdsignal_fix', 'macdhist_fix', 'mfi', 'minus_di', 'mom', 'plus_di', 'dm',
            'ppo', 'roc', 'rocp', 'rocr', 'rocr100', 'rsi', 'slowk', 'slowd', 'fastk', 'fastd',
            'fastk_rsi', 'fastd_rsi', 'trix', 'ultosc', 'willr', 'ad', 'adosc', 'obv', 'atr', 'natr',
            'ht_dcperiod', 'ht_dcphase', 'ht_phasor_inphase', 'ht_phasor_quadrature',
            'ht_sine_sine', 'ht_sine_leadsine', 'ht_trendmod', 'avgprice', 'medprice', 'typprice', 'wclprice'
        ]
        self.TECHNICAL_PATTERNS = [g for g in self.df_full.keys() if g.find('cdl') >= 0]

        self.NUMERICAL = (
            self.GROWTH + self.TECHNICAL_INDICATORS + self.TECHNICAL_PATTERNS +
            self.CUSTOM_NUMERICAL + self.MACRO
        )

        self.OTHER = [k for k in self.df_full.keys()
                      if k not in self.OHLCV + self.CATEGORICAL + self.NUMERICAL +
                      self.TO_DROP + self.TO_PREDICT]

    def _define_dummies(self):
        self.df_full.loc[:, 'Month'] = pd.to_datetime(self.df_full.Month_x, errors='coerce').dt.strftime('%B')
        self.df_full['Weekday'] = self.df_full['Weekday'].astype(str)
        self.df_full['Month_Weekday'] = self.df_full['Month'] + '_' + self.df_full['Weekday']
        self.CATEGORICAL.append('Month_Weekday')

        dummy_variables = pd.get_dummies(self.df_full[self.CATEGORICAL], dtype='int32')
        self.df_full = pd.concat([self.df_full, dummy_variables], axis=1)
        self.DUMMIES = dummy_variables.keys().to_list()

    # --------------------------
    # Temporal split
    # --------------------------
    def _perform_temporal_split(self, df, min_date, max_date, train_prop=0.7, val_prop=0.15):
        train_end = min_date + pd.Timedelta(days=(max_date - min_date).days * train_prop)
        val_end = train_end + pd.Timedelta(days=(max_date - min_date).days * val_prop)

        split_labels = []
        for date in df['Date']:
            if date <= train_end:
                split_labels.append('train')
            elif date <= val_end:
                split_labels.append('validation')
            else:
                split_labels.append('test')

        df['split'] = split_labels
        return df

    def _clean_dataframe_from_inf_and_nan(self, df):
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df

    # --------------------------
    # ML dataframes
    # --------------------------
    def _define_dataframes_for_ML(self):
        features_list = self.NUMERICAL + self.DUMMIES
        to_predict = ['is_positive_growth_1h_future', 'is_positive_growth_4h_future']

        self.train_df = self.df_full[self.df_full.split == 'train'].copy()
        self.valid_df = self.df_full[self.df_full.split == 'validation'].copy()
        self.train_valid_df = self.df_full[self.df_full.split.isin(['train', 'validation'])].copy()
        self.test_df = self.df_full[self.df_full.split == 'test'].copy()

        def split_xy(df):
            X = df[features_list].copy()
            y = df[to_predict].copy()
            return self._clean_dataframe_from_inf_and_nan(X), y

        self.X_train, self.y_train = split_xy(self.train_df)
        self.X_valid, self.y_valid = split_xy(self.valid_df)
        self.X_train_valid, self.y_train_valid = split_xy(self.train_valid_df)
        self.X_test, self.y_test = split_xy(self.test_df)
        self.X_all, self.y_all = split_xy(self.df_full)

        print(f'X_train {self.X_train.shape}, X_valid {self.X_valid.shape}, '
              f'X_test {self.X_test.shape}, X_all {self.X_all.shape}')

    def prepare_dataframe(self):
        print("Preparing dataframe...")
        self._define_feature_sets()
        self._define_dummies()
        min_date, max_date = self.df_full.Date.min(), self.df_full.Date.max()
        self._perform_temporal_split(self.df_full, min_date, max_date)
        self._define_dataframes_for_ML()

    # --------------------------
    # Training
    # --------------------------
    def train_xgboost(self):
        print("Training separate XGBoost models for 1h and 4h...")

        params = dict(
            colsample_bytree=0.49940025099516744,
            gamma=0.8770610350651781,
            learning_rate=0.012070944572937612,
            max_depth=4,
            min_child_weight=28.663297628995185,
            n_estimators=500,
            eval_metric='auc',
            n_jobs=-1,
            objective='binary:logistic',
            random_state=42,
        )

        targets = {
            "1h": "is_positive_growth_1h_future",
            "4h": "is_positive_growth_4h_future"
        }

        for horizon, target in targets.items():
            print(f"→ Training {horizon} model...")
            model = xgb.XGBClassifier(**params)
            model.fit(self.X_train_valid, self.y_train_valid[target])
            self.models[horizon] = model

    def persist(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        for horizon, model in self.models.items():
            model_filename = f"xgboost_model_{horizon}.joblib"
            joblib.dump(model, os.path.join(data_dir, model_filename))

    def load(self, data_dir: str):
        self.models = {}
        for horizon in ["1h", "4h"]:
            path = os.path.join(data_dir, f"xgboost_model_{horizon}.joblib")
            if os.path.exists(path):
                self.models[horizon] = joblib.load(path)

    # --------------------------
    # Inference
    # --------------------------
    def make_inference(self):
        for horizon, model in self.models.items():
            pred_col = f"pred_xgboost_{horizon}_best"
            prob_col = f"prob_pred_xgboost_{horizon}_best"
            rank_col = f"pred_xgboost_{horizon}_best_rank"

            y_pred_prob = model.predict_proba(self.X_all)[:, 1]
            y_pred_class = (y_pred_prob >= 0.56).astype(int)

            self.df_full[pred_col] = y_pred_class.astype(bool)
            self.df_full[prob_col] = y_pred_prob
            self.df_full[rank_col] = self.df_full.groupby("Date")[prob_col].rank(method="first", ascending=False)

            self.prediction_cols.extend([pred_col, prob_col, rank_col])

        out_path = os.path.join("local_data", "tickers_df.parquet")
        os.makedirs("local_data", exist_ok=True)
        self.df_full.to_parquet(out_path, compression="brotli")
        print(f"Saved predictions (1h & 4h) to {out_path}")
