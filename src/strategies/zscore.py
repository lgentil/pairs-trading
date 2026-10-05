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

class DynamicZScoreStrategy(BasePairsStrategy):
    """Executes a pairs trading mean-reversion strategy based on normalized Spread Z-Scores.

    Attributes:
        df (pd.DataFrame): DataFrame containing price and spread series.
        ticker_x (str): Ticker symbol for Asset X.
        ticker_y (str): Ticker symbol for Asset Y.
        entry_z (float): Z-score threshold for opening long/short positions.
        exit_z (float): Z-score threshold for taking profit/reverting to mean.
        stop_mult (Optional[float]): Multiplier relative to entry_z for stop-loss execution.
        stop_loss_z (Optional[float]): Absolute Z-score threshold derived from stop_mult.
        lookback_window (Union[int, str]): Normalization window ('dynamic' or integer).
        transaction_cost (float): Percentage cost applied per trade leg.
        compounding (bool): Determines if returns are reinvested (True) or fixed size (False).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        ticker_x: str,
        ticker_y: str,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        stop_mult: Optional[float] = None,
        stop_loss_z: Optional[float] = None,
        lookback_window: Union[int, str] = 'dynamic',
        transaction_cost: float = 0.0,
        compounding: bool = False,
        **kwargs: Any
    ) -> None:
        """Initializes the Dynamic Z-Score model and runs the execution pipeline.

        Args:
            df (pd.DataFrame): Input market data DataFrame.
            ticker_x (str): Ticker symbol for Asset X.
            ticker_y (str): Ticker symbol for Asset Y.
            entry_z (float, optional): Entry critical value. Defaults to 2.0.
            exit_z (float, optional): Exit critical value. Defaults to 0.0.
            stop_mult (Optional[float], optional): Multiplier of entry_z for stop loss. Defaults to None.
            stop_loss_z (Optional[float], optional): Absolute stop loss critical value. Defaults to None.
            lookback_window (Union[int, str], optional): Lookback window type. Defaults to 'dynamic'.
            transaction_cost (float, optional): Execution slippage and commission per trade. Defaults to 0.0.
            compounding (bool, optional): Compounding method for equity curve. Defaults to True.
            **kwargs: Extra unused keyword parameters.
        """
        self.entry_z: float = float(entry_z)
        self.exit_z: float = float(exit_z)
        self.lookback_window: Union[int, str] = lookback_window
        self.transaction_cost: float = float(transaction_cost)
        self.compounding: bool = bool(compounding)
        
        self.stop_mult: Optional[float] = float(stop_mult) if (stop_mult is not None and not pd.isna(stop_mult)) else None
        
        if self.stop_mult is not None:
            self.stop_loss_z: Optional[float] = float(self.entry_z * self.stop_mult)
        elif stop_loss_z is not None and not pd.isna(stop_loss_z):
            self.stop_loss_z: Optional[float] = float(stop_loss_z)
        else:
            self.stop_loss_z = None

        super().__init__(
            df,
            ticker_x,
            ticker_y,
            entry_z=self.entry_z,
            exit_z=self.exit_z,
            stop_mult=self.stop_mult,
            stop_loss_z=self.stop_loss_z,
            lookback_window=self.lookback_window,
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
        if self.lookback_window == 'dynamic':
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
            fixed_window = int(self.lookback_window)
            self.df['spread_mean'] = self.df['Spread'].rolling(window=fixed_window).mean()
            self.df['spread_std'] = self.df['Spread'].rolling(window=fixed_window).std(ddof=1)

        self.df['z_score'] = (self.df['Spread'] - self.df['spread_mean']) / self.df['spread_std']

        z_vals = self.df['z_score'].values
        hr_array = self.df['Hedge_Ratio'].values
        n = len(self.df)
        
        positions = np.zeros(n)
        exec_hr = np.full(n, np.nan)
        current_pos = 0.0
        current_hr = np.nan
        stopped_out = False

        ez = self.entry_z
        xz = self.exit_z
        sz = self.stop_loss_z if self.stop_loss_z is not None else np.inf

        for i in range(n):
            z = z_vals[i]

            if np.isnan(z):
                positions[i] = current_pos
                exec_hr[i] = current_hr
                continue

            if current_pos == 1.0:
                if z >= -xz:
                    current_pos = 0.0
                    current_hr = np.nan
                elif z < -sz:
                    current_pos = 0.0
                    current_hr = np.nan
                    stopped_out = True
                    
            elif current_pos == -1.0:
                if z <= xz:
                    current_pos = 0.0
                    current_hr = np.nan
                elif z > sz:
                    current_pos = 0.0
                    current_hr = np.nan
                    stopped_out = True

            if stopped_out and i > 0:
                z_prev = z_vals[i - 1]
                if (z >= 0.0 and z_prev < 0.0) or (z <= 0.0 and z_prev > 0.0):
                    stopped_out = False

            if current_pos == 0.0 and not stopped_out:
                if z < -ez and z >= -sz:
                    current_pos = 1.0
                    current_hr = hr_array[i]
                elif z > ez and z <= sz:
                    current_pos = -1.0
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
