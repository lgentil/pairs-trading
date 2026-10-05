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
from .base import BasePairsStrategy

class DynamicMeanReversionStrategy(BasePairsStrategy):
    """Executes a pairs trading mean-reversion strategy based on Bollinger Bands.

    Attributes:
        df (pd.DataFrame): DataFrame containing price and spread series.
        ticker_x (str): Ticker symbol for Asset X.
        ticker_y (str): Ticker symbol for Asset Y.
        num_std (float): Number of standard deviations for the entry bands.
        stop_mult (Optional[float]): Multiplier relative to num_std for stop-loss execution.
        ma_window (Union[int, str]): Window for calculating moving average and volatility.
        transaction_cost (float): Percentage cost applied per trade leg.
        compounding (bool): Determines if returns are reinvested (True) or fixed size (False).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ticker_x: str,
        ticker_y: str,
        num_std: float = 2.0,
        stop_mult: Optional[float] = None,
        ma_window: Union[int, str] = 'dynamic',
        transaction_cost: float = 0.0,
        compounding: bool = False,
        **kwargs: Any
    ) -> None:
        """Initializes the Dynamic Mean Reversion model.

        Args:
            df (pd.DataFrame): Input market data DataFrame.
            ticker_x (str): Ticker symbol for Asset X.
            ticker_y (str): Ticker symbol for Asset Y.
            num_std (float, optional): Entry band standard deviation multiplier. Defaults to 2.0.
            stop_mult (Optional[float], optional): Stop loss multiplier. Defaults to None.
            ma_window (Union[int, str], optional): Lookback window type. Defaults to 'dynamic'.
            transaction_cost (float, optional): Execution slippage and commission per trade leg. Defaults to 0.0.
            compounding (bool, optional): Compounding method for equity curve. Defaults to True.
            **kwargs: Extra unused keyword parameters.
        """
        self.num_std: float = float(num_std)
        self.stop_mult: Optional[float] = float(stop_mult) if stop_mult is not None else None
        self.ma_window: Union[int, str] = ma_window
        self.transaction_cost: float = float(transaction_cost)
        self.compounding: bool = bool(compounding)

        super().__init__(
            df,
            ticker_x,
            ticker_y,
            num_std=self.num_std,
            stop_mult=self.stop_mult,
            ma_window=self.ma_window,
            transaction_cost=self.transaction_cost,
            compounding=self.compounding,
            **kwargs
        )
        self.run()

    def run(self) -> pd.DataFrame:
        """Calculates indicators, signals, positions, transaction costs, and strategy returns.

        Returns:
            pd.DataFrame: Calculated DataFrame with cumulative net returns.
        """
        if self.ma_window == 'dynamic':
            ma = np.full(len(self.df), np.nan)
            std_dev = np.full(len(self.df), np.nan)
            spread_vals = self.df['Spread'].values
            lookback_vals = self.df['Dynamic_Lookback'].values

            for i in range(len(self.df)):
                lb = lookback_vals[i]
                if pd.notnull(lb) and not np.isnan(lb):
                    lb = int(lb)
                    if i >= lb - 1 and lb > 1:
                        window_slice = spread_vals[i - lb + 1 : i + 1]
                        ma[i] = np.mean(window_slice)
                        std_dev[i] = np.std(window_slice, ddof=1)

            self.df['spread_mean'] = ma
            self.df['spread_std'] = std_dev
        else:
            fixed_window = int(self.ma_window)
            self.df['spread_mean'] = self.df['Spread'].rolling(window=fixed_window).mean()
            self.df['spread_std'] = self.df['Spread'].rolling(window=fixed_window).std(ddof=1)

        self.df['upper_band'] = self.df['spread_mean'] + (self.num_std * self.df['spread_std'])
        self.df['lower_band'] = self.df['spread_mean'] - (self.num_std * self.df['spread_std'])

        if self.stop_mult is not None:
            self.df['stop_upper'] = self.df['spread_mean'] + (self.num_std * self.stop_mult * self.df['spread_std'])
            self.df['stop_lower'] = self.df['spread_mean'] - (self.num_std * self.stop_mult * self.df['spread_std'])
        else:
            self.df['stop_upper'] = np.inf
            self.df['stop_lower'] = -np.inf

        spread_vals = self.df['Spread'].values
        ma_vals = self.df['spread_mean'].values
        ub_vals = self.df['upper_band'].values
        lb_vals = self.df['lower_band'].values
        su_vals = self.df['stop_upper'].values
        sl_vals = self.df['stop_lower'].values
        hr_array = self.df['Hedge_Ratio'].values

        n = len(self.df)
        positions = np.zeros(n)
        exec_hr = np.full(n, np.nan)

        current_pos = 0.0
        current_hr = np.nan
        stopped_out = False

        for i in range(n):
            s = spread_vals[i]
            mean_val = ma_vals[i]
            ub = ub_vals[i]
            lb = lb_vals[i]
            su = su_vals[i]
            sl = sl_vals[i]

            if np.isnan(s) or np.isnan(ub):
                positions[i] = current_pos
                exec_hr[i] = current_hr
                continue

            if current_pos == -1.0:
                if s <= mean_val:
                    current_pos = 0.0
                    current_hr = np.nan
                elif s >= su:
                    current_pos = 0.0
                    current_hr = np.nan
                    stopped_out = True
            elif current_pos == 1.0:
                if s >= mean_val:
                    current_pos = 0.0
                    current_hr = np.nan
                elif s <= sl:
                    current_pos = 0.0
                    current_hr = np.nan
                    stopped_out = True

            if stopped_out and i > 0:
                s_prev = spread_vals[i - 1]
                mean_prev = ma_vals[i - 1]
                if (s >= mean_val and s_prev < mean_prev) or (s <= mean_val and s_prev > mean_prev):
                    stopped_out = False

            if current_pos == 0.0 and not stopped_out:
                if s > ub and s < su:
                    current_pos = -1.0
                    current_hr = hr_array[i]
                elif s < lb and s > sl:
                    current_pos = 1.0
                    current_hr = hr_array[i]

            positions[i] = current_pos
            exec_hr[i] = current_hr

        self.df['positions'] = positions
        self.df['trade_hr'] = exec_hr

        prev_hr = self.df['trade_hr'].shift(1)
        prev_pos = self.df['positions'].shift(1)
        
        delta_y = self.df[self.ticker_y] - self.df[self.ticker_y].shift(1)
        delta_x = self.df[self.ticker_x] - self.df[self.ticker_x].shift(1)
        
        pnl_cesta = delta_y - (prev_hr * delta_x)
        capital_alocado = self.df[self.ticker_y].shift(1).abs() + (prev_hr.abs() * self.df[self.ticker_x].shift(1).abs())
        
        self.df['percentage_change'] = np.where(
            capital_alocado != 0, 
            pnl_cesta / capital_alocado, 
            0.0
        )
        
        self.df['strategy_returns'] = prev_pos * self.df['percentage_change']
        
        position_diff = self.df['positions'].diff().fillna(0.0).abs()
        self.df['transaction_costs'] = position_diff * self.transaction_cost
        self.df['net_strategy_returns'] = self.df['strategy_returns'].fillna(0.0) - self.df['transaction_costs']
        
        if self.compounding:
            self.df['cumulative_returns'] = (self.df['net_strategy_returns'] + 1.0).cumprod()
        else:
            self.df['cumulative_returns'] = self.df['net_strategy_returns'].cumsum() + 1.0

        return self.df
