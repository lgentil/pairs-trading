# Import libraries
# Python version 3.14.4

import datetime as dt
import itertools
import math
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from decimal import Decimal
import plotly.express as px
import matplotlib.patches as mpatches
import quantstats as qs
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, coint
import yfinance as yf
import requests

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

class RollingSpreadAnalyzer:
    """Performs rolling cointegration and dynamic spread analysis on a pair of financial assets.

    Calculates dynamic parameters over a specified rolling window. Computes the dynamic 
    Hedge Ratio using Rolling OLS, derives the dynamic spread time series, iteratively 
    computes the Half-Life of mean reversion via linear algebra, and performs rolling 
    stationarity checks via the Augmented Dickey-Fuller (ADF) test.

    Attributes:
        df (pd.DataFrame): DataFrame containing the raw price data.
        ticker_x (str): Ticker symbol for the independent asset (Asset X).
        ticker_y (str): Ticker symbol for the dependent asset (Asset Y).
        window (int): Rolling estimation lookback window size. Defaults to 90.
    """

    def __init__(self, df: pd.DataFrame, ticker_x: str, ticker_y: str, window: int = 90) -> None:
        """Initializes the RollingSpreadAnalyzer with asset series and lookback parameters.

        Args:
            df (pd.DataFrame): DataFrame containing asset price data.
            ticker_x (str): Ticker symbol for the independent asset.
            ticker_y (str): Ticker symbol for the dependent asset.
            window (int, optional): Rolling lookback window size. Defaults to 90.
        """
        self.df: pd.DataFrame = df.copy()
        self.ticker_x: str = ticker_x
        self.ticker_y: str = ticker_y
        self.window: int = window

    def run_analysis(self) -> pd.DataFrame:
        """Executes the rolling parameter calculation pipeline.

        Returns:
            pd.DataFrame: DataFrame containing the dynamic Hedge Ratio, Spread,
            ADF p-values, stationarity flags, Half-Life, and dynamic lookbacks.
        """
        # 1. Rolling OLS estimation for dynamic Hedge Ratio (HR)
        endog = self.df[self.ticker_y]
        exog = self.df[self.ticker_x]

        rolling_model = RollingOLS(endog, exog, window=self.window)
        rolling_res = rolling_model.fit(params_only=True)

        self.df['Hedge_Ratio'] = rolling_res.params[self.ticker_x]
        self.df['Spread'] = self.df[self.ticker_y] - (self.df['Hedge_Ratio'] * self.df[self.ticker_x])

        y_arr = endog.values
        x_arr = exog.values
        hr_arr = self.df['Hedge_Ratio'].values

        hl_list = np.full(len(self.df), np.nan)
        adf_pvalue_list = np.full(len(self.df), np.nan)
        is_stat_list = np.full(len(self.df), False)

        # 2. Iterate across windows for stationarity and half-life calculations
        for i in range(self.window - 1, len(self.df)):
            hr = hr_arr[i]
            if np.isnan(hr):
                continue

            y_slice = y_arr[i - self.window + 1 : i + 1]
            x_slice = x_arr[i - self.window + 1 : i + 1]
            spread_slice = y_slice - (hr * x_slice)

            # ADF Stationarity Test
            try:
                adf_res = adfuller(spread_slice, maxlag=1)
                p_value = adf_res[1]
                adf_pvalue_list[i] = p_value
                is_stat_list[i] = p_value < 0.10
            except Exception:
                pass

            # Half-Life Calculation via Ornstein-Uhlenbeck Linear Formulation
            mean_spread = np.mean(spread_slice)
            dev_from_mean = mean_spread - spread_slice[:-1]
            diff_spread = spread_slice[1:] - spread_slice[:-1]

            variance_x = np.sum(dev_from_mean ** 2)
            if variance_x > 0:
                theta = np.sum(dev_from_mean * diff_spread) / variance_x
                if theta > 0:
                    hl_list[i] = math.log(2) / theta

        # 3. Store calculated metrics
        self.df['ADF_pvalue'] = adf_pvalue_list
        self.df['Is_Stationary'] = is_stat_list
        self.df['Half_Life'] = hl_list
        self.df['Dynamic_Lookback'] = self.df['Half_Life'].apply(
            lambda x: min(max(int(round(x)), 5), 100) if pd.notnull(x) else np.nan
        )

        return self.df
