import numpy as np

import pandas as pd
import yfinance as yf

from tqdm import tqdm
from datetime import datetime, timedelta

import time
import os
import pandas_datareader as pdr


# https://finance.yahoo.com/markets/currencies/

CURRENCY_PAIRS = [
    'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X', 'EURJPY=X', 'GBPJPY=X',
	'EURGBP=X', 'EURCAD=X', 'EURSEK=X', 'EURCHF=X', 'EURHUF=X', 'CNY=X', 'HKD=X',
	'SGD=X', 'INR=X', 'MXN=X', 'PHP=X', 'IDR=X', 'THB=X', 'MYR=X', 'ZAR=X', 'RUB=X'
]


class DataRepository:
  ticker_df: pd.DataFrame
  indexes_df: pd.DataFrame
  macro_df: pd.DataFrame

  #min_date: str
  ALL_TICKERS: list[str] = CURRENCY_PAIRS        # 23 Currency Pairs

  def __init__(self):
    self.ticker_df = None
    self.indexes_df = None
    self.macro_df = None

  def _get_growth_df(self, df:pd.DataFrame, prefix:str)->pd.DataFrame:
    '''Help function to produce a df with growth columns'''
    for i in [1,4,7,10,15]:
      df['growth_'+prefix+'_'+str(i)+'h'] = df['Close'] / df['Close'].shift(i)
      GROWTH_KEYS = [k for k in df.keys() if k.startswith('growth')]
    return df[GROWTH_KEYS]
  
  def _fetch_index_with_fallback(self, yf_symbol, name, period='max'):
    '''Fetch index data with yfinance'''
    data = None
    # Try Yahoo Finance API
    try:
      ticker_obj = yf.Ticker(yf_symbol)
      data = ticker_obj.history(period=period, interval="1h")
      if not data.empty:
        print(f"yfinance SUCCESS: Got {len(data)} rows for {name}")
        data.index = pd.to_datetime(data.index)
      else:
        print(f"yfinance returned no data for {name}")
    except Exception as e:
      print(f"yfinance error for {name}: {e}")
    # If no data, create empty DataFrame with proper structure
    if data is None or data.empty:
      print(f"No data available for {name} from any source, creating empty DataFrame")
      data = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
      data.index = pd.DatetimeIndex([])
    # Sleep to avoid overloading API
    time.sleep(1)
    return data
    
  def fetch(self, period = None):
      '''Fetch all data from APIs'''

      print('Fetching Tickers info from YFinance')
      self.fetch_tickers(period=period)
      print('Fetching Indexes info from YFinance')
      self.fetch_indexes(period=period)
      print('Fetching Macro info from FRED (Pandas_datareader)')
      self.fetch_macro(min_date=period)
  
  def fetch_tickers(self, period='max'):
    '''Fetch Tickers data from yfinance API with Stooq fallback for REAL data'''   

    print(f'Going download data for this tickers: {self.ALL_TICKERS}')
    tq = tqdm(self.ALL_TICKERS)
    
    for i,ticker in enumerate(tq):
      tq.set_description(ticker)
      historyPrices = None

      # First try: Use the 2025 colab approach with yfinance
      try:
        ticker_obj = yf.Ticker(ticker)
        historyPrices = ticker_obj.history(period=period, interval="1h")
        
        if not historyPrices.empty:
          print(f"yfinance SUCCESS: Got {len(historyPrices)} rows for {ticker}")
        else:
          print(f"yfinance returned no data for {ticker}")
          
      except Exception as e:
        print(f"yfinance error for {ticker}: {e}")
      
      # Skip if no data from either source
      if historyPrices is None or historyPrices.empty:
        print(f"No data available for {ticker} from YFinance, skipping...")
        time.sleep(1)
        continue

      # Ensure index is datetime
      historyPrices.index = pd.to_datetime(historyPrices.index)

      # generate features for historical prices, and what we want to predict
      historyPrices['Ticker'] = ticker
      historyPrices['Year']= historyPrices.index.year
      historyPrices['Month'] = historyPrices.index.month
      historyPrices['Weekday'] = historyPrices.index.weekday
      historyPrices['Date'] = historyPrices.index.date
      
     # historical returns
      for i in [1,4,7,10,15]:
        historyPrices['growth_'+str(i)+'h'] = historyPrices['Close'] / historyPrices['Close'].shift(i)
    # future returns
      historyPrices['growth_future_1h'] = historyPrices['Close'].shift(-1) / historyPrices['Close']

      # Technical indicators
      # SimpleMovingAverage 10 days and 20 days
      historyPrices['SMA10']= historyPrices['Close'].rolling(10).mean()
      historyPrices['SMA20']= historyPrices['Close'].rolling(20).mean()
      historyPrices['growing_moving_average'] = np.where(historyPrices['SMA10'] > historyPrices['SMA20'], 1, 0)
      historyPrices['high_minus_low_relative'] = (historyPrices['High'] - historyPrices['Low']) / historyPrices['Close']
      
     # 30d rolling volatility : https://ycharts.com/glossary/terms/rolling_vol_30
      historyPrices['volatility'] =   historyPrices['growth_1h'].rolling(30).std() * np.sqrt(252*24)

    # what we want to predict
      historyPrices['is_positive_growth_1h_future'] = np.where(historyPrices['growth_future_1h'] > 1, 1, 0)

      # sleep 1 sec between downloads - not to overload the API server
      time.sleep(1)

      if self.ticker_df is None:
        self.ticker_df = historyPrices
      else:
        self.ticker_df = pd.concat([self.ticker_df, historyPrices], ignore_index=True)

      
  def fetch_indexes(self, period='max'):
    '''Fetch Indexes data from yfinance '''
    # Define index mappings: (yfinance_symbol, name)
    index_mappings = [
      ("^GDAXI", "DAX"),
      ("^GSPC", "S&P500"),
      ("^DJI", "Dow Jones"),
      ("^VIX", "VIX"),
      ("GC=F", "Gold"),
      ("CL=F", "WTI Oil"),
      ("BZ=F", "Brent Oil"),
      ("BTC-USD", "Bitcoin")
    ]

    # Fetch each index with fallback (including FRED symbols for missing data)
    dax_hourly = self._fetch_index_with_fallback("^GDAXI", "DAX", period)
    snp500_hourly = self._fetch_index_with_fallback("^GSPC", "S&P500", period)
    dji_hourly = self._fetch_index_with_fallback("^DJI", "Dow Jones", period)
    vix = self._fetch_index_with_fallback("^VIX", "VIX", period)
    gold = self._fetch_index_with_fallback("GC=F", "Gold", period)
    crude_oil = self._fetch_index_with_fallback("CL=F", "WTI Oil", period)
    brent_oil = self._fetch_index_with_fallback("BZ=F", "Brent Oil", period)
    btc_usd = self._fetch_index_with_fallback("BTC-USD", "Bitcoin", period)

    # Prepare to merge
    dax_hourly_to_merge = self._get_growth_df(dax_hourly, 'dax')
    snp500_to_merge = self._get_growth_df(snp500_hourly, 'snp500')
    dji_hourly_to_merge = self._get_growth_df(dji_hourly, 'dji')
    vix_to_merge = self._get_growth_df(vix, 'vix')
    gold_to_merge = self._get_growth_df(gold, 'gold')
    crude_oil_to_merge = self._get_growth_df(crude_oil,'wti_oil')
    brent_oil_to_merge = self._get_growth_df(brent_oil,'brent_oil')
    btc_usd_to_merge = self._get_growth_df(btc_usd,'btc_usd')

    # Merging
    m2 = pd.merge(snp500_to_merge,
          dax_hourly_to_merge,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    m3 = pd.merge(m2,
          dji_hourly_to_merge,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    m4 = pd.merge(m3,
          vix_to_merge,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    m5 = pd.merge(m4,
          gold_to_merge,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    m6 = pd.merge(m5,
          crude_oil_to_merge,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    m7 = pd.merge(m6,
          brent_oil_to_merge,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    m8 = pd.merge(m7,
          btc_usd_to_merge,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    self.indexes_df = m8

  def fetch_macro(self, min_date=None):
    '''Fetch Macro data from FRED (using Pandas datareader)'''
    if min_date is None:
      min_date = "2023-08-27"
    else:
      min_date = pd.to_datetime(min_date)

    # Real Potential Gross Domestic Product (GDPPOT), Billions of Chained 2012 Dollars, QUARTERLY
    gdppot = pdr.DataReader("GDPPOT", "fred", start=min_date)
    gdppot['gdppot_us_yoy'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(4)-1
    gdppot['gdppot_us_qoq'] = gdppot.GDPPOT/gdppot.GDPPOT.shift(1)-1
    time.sleep(1)

    cpilfesl = pdr.DataReader("CPILFESL", "fred", start=min_date)
    cpilfesl['cpi_core_yoy'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(12)-1
    cpilfesl['cpi_core_mom'] = cpilfesl.CPILFESL/cpilfesl.CPILFESL.shift(1)-1    
    time.sleep(1)

    trade_balance = pdr.DataReader("BOPGSTB", "fred", start=min_date)
    trade_balance['trade_balance_us_yoy'] = trade_balance.BOPGSTB/trade_balance.BOPGSTB.shift(12)-1
    trade_balance['trade_balance_us_mom'] = trade_balance.BOPGSTB/trade_balance.BOPGSTB.shift(1)-1

    fedfunds = pdr.DataReader("FEDFUNDS", "fred", start=min_date)
    time.sleep(1)

    dgs1 = pdr.DataReader("DGS1", "fred", start=min_date)
    time.sleep(1)

    dgs5 = pdr.DataReader("DGS5", "fred", start=min_date)
    time.sleep(1)

    dgs10 = pdr.DataReader("DGS10", "fred", start=min_date)
    time.sleep(1)

    gdppot_to_merge = gdppot[['gdppot_us_yoy','gdppot_us_qoq']]
    cpilfesl_to_merge = cpilfesl[['cpi_core_yoy','cpi_core_mom']]
    trade_balance_to_merge = trade_balance[['trade_balance_us_yoy','trade_balance_us_mom']]

    # Merging - start from daily stats (dgs1)
    m2 = pd.merge(dgs1,
          dgs5,
          left_index=True,
          right_index=True,
          how='left',
          validate='one_to_one')

    m2['Date'] = m2.index
    m2['Quarter'] = m2.Date.dt.to_period('Q').dt.to_timestamp()

    m3 = pd.merge(m2,
          gdppot_to_merge,
          left_on='Quarter',
          right_index=True,
          how='left',
          validate='many_to_one')

    m3['Month'] = m2.Date.dt.to_period('M').dt.to_timestamp()

    m4 = pd.merge(m3,
          cpilfesl_to_merge,
          left_on='Month',
          right_index=True,
          how='left',
          validate='many_to_one')

    m5 = pd.merge(m4,
          trade_balance_to_merge,
          how='left',
          left_on='Month',
          right_index=True,
          validate = "many_to_one")

    m6 = pd.merge(m5,
          fedfunds,
          left_on='Month',
          right_index=True,
          how='left',
          validate='many_to_one')

    m7 = pd.merge(m6,
          dgs10,
          left_on='Date',
          right_index=True,
          how='left',
          validate='one_to_one')

    fields_to_fill = ['cpi_core_yoy', 'cpi_core_mom', 'FEDFUNDS', 'DGS1', 'DGS5', 'DGS10']
    for field in fields_to_fill:
      m7[field] = m7[field].ffill()

    self.macro_df =  m7   

  def persist(self, data_dir:str):
    '''Save dataframes to files in a local directory 'dir' '''
    os.makedirs(data_dir, exist_ok=True)

    # Only save if dataframes exist and are not empty
    if self.ticker_df is not None and not self.ticker_df.empty:
      file_name = 'tickers_df.parquet'
      if os.path.exists(os.path.join(data_dir, file_name)):
        os.remove(os.path.join(data_dir, file_name))
      self.ticker_df.to_parquet(os.path.join(data_dir,file_name), compression='brotli')
      print(f"Saved {len(self.ticker_df)} ticker records")
    else:
      print("No ticker data to save")
  
    if self.indexes_df is not None and not self.indexes_df.empty:
      file_name = 'indexes_df.parquet'
      if os.path.exists(os.path.join(data_dir, file_name)):
        os.remove(os.path.join(data_dir, file_name))
      self.indexes_df.to_parquet(os.path.join(data_dir,file_name), compression='brotli')
      print(f"Saved {len(self.indexes_df)} index records")
    else:
      print("No index data to save")
  
    if self.macro_df is not None and not self.macro_df.empty:
      file_name = 'macro_df.parquet'
      if os.path.exists(os.path.join(data_dir, file_name)):
        os.remove(os.path.join(data_dir, file_name))
      self.macro_df.to_parquet(os.path.join(data_dir,file_name), compression='brotli')
      print(f"Saved {len(self.macro_df)} macro records")
    else:
      print("No macro data to save")

  def load(self, data_dir:str):
    """Load files from the local directory"""
    self.ticker_df = pd.read_parquet(os.path.join(data_dir,'tickers_df.parquet'))
    self.macro_df = pd.read_parquet(os.path.join(data_dir,'macro_df.parquet'))
    self.indexes_df = pd.read_parquet(os.path.join(data_dir,'indexes_df.parquet'))






