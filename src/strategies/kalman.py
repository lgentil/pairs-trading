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

class KalmanZScoreStrategy(BasePairsStrategy):
    """Executes a dynamic pairs trading strategy using the Kalman Filter with optional stationarity diagnostics.

    Estimates time-varying Hedge Ratio (slope) and Intercept via recursive state-space updates.
    Generates standardized trading signals based on the innovation error (spread) normalized by
    the dynamic innovation variance (Q_t).

    Attributes:
        df (pd.DataFrame): DataFrame containing raw price series for Asset X and Asset Y.
        ticker_x (str): Ticker symbol for Asset X.
        ticker_y (str): Ticker symbol for Asset Y.
        entry_z (float): Z-score threshold for opening long/short positions.
        exit_z (float): Z-score threshold for reverting to mean / closing positions.
        stop_loss_z (Optional[float]): Z-score threshold for stop loss.
        delta (float): Process noise transition covariance parameter.
        obs_variance (float): Observation noise variance (R).
        adf_window (int): Rolling lookback window for ADF stationarity validation.
        compute_stationarity (bool): Flag to conditionally execute the ADF loop only when required.
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
        stop_loss_z: Optional[float] = 3.0,
        delta: float = 1e-4,
        obs_variance: float = 1e-3,
        adf_window: int = 90,
        compute_stationarity: bool = False,
        transaction_cost: float = 0.0,
        compounding: bool = False,
        **kwargs: Any
    ) -> None:
        """Initializes the Kalman Filter Strategy and executes the calculation pipeline.

        Args:
            df (pd.DataFrame): Market data DataFrame.
            ticker_x (str): Ticker symbol for Asset X.
            ticker_y (str): Ticker symbol for Asset Y.
            entry_z (float, optional): Entry critical value. Defaults to 2.0.
            exit_z (float, optional): Exit critical value. Defaults to 0.0.
            stop_loss_z (Optional[float], optional): Stop loss critical value. Defaults to 3.0.
            delta (float, optional): State covariance transition speed. Defaults to 1e-4.
            obs_variance (float, optional): Measurement noise variance. Defaults to 1e-3.
            adf_window (int, optional): Rolling lookback window for ADF test. Defaults to 90.
            compute_stationarity (bool, optional): Whether to run the rolling ADF loop. Defaults to False.
            transaction_cost (float, optional): Execution slippage and commission per trade. Defaults to 0.0.
            compounding (bool, optional): Compounding method for equity curve. Defaults to False.
            **kwargs: Extra unused keyword parameters.
        """
        self.entry_z: float = float(entry_z)
        self.exit_z: float = float(exit_z)
        self.stop_loss_z: Optional[float] = float(stop_loss_z) if stop_loss_z is not None else None
        self.delta: float = float(delta)
        self.obs_variance: float = float(obs_variance)
        self.adf_window: int = int(adf_window)
        self.compute_stationarity: bool = compute_stationarity
        self.transaction_cost: float = float(transaction_cost)
        self.compounding: bool = bool(compounding)

        super().__init__(
            df,
            ticker_x,
            ticker_y,
            entry_z=self.entry_z,
            exit_z=self.exit_z,
            stop_loss_z=self.stop_loss_z,
            delta=self.delta,
            obs_variance=self.obs_variance,
            adf_window=self.adf_window,
            compute_stationarity=self.compute_stationarity,
            transaction_cost=self.transaction_cost,
            compounding=self.compounding,
            **kwargs
        )
        self.run()

    def run(self) -> pd.DataFrame:
        """Executes the Kalman state-space recursive estimation, signals, regimes, and returns.

        Returns:
            pd.DataFrame: Calculated DataFrame with Kalman metrics, stationarity flags, and returns.
        """
        x = self.df[self.ticker_x].values
        y = self.df[self.ticker_y].values
        n = len(self.df)

        state = np.zeros(2)
        state_cov = np.eye(2)
        Vw = self.delta / (1.0 - self.delta) * np.eye(2)
        Ve = self.obs_variance

        hedge_ratios = np.zeros(n)
        intercepts = np.zeros(n)
        spread_errors = np.zeros(n)
        innovation_vars = np.zeros(n)
        z_scores = np.zeros(n)

        for t in range(n):
            H = np.array([1.0, x[t]])
            state_cov_prior = state_cov + Vw
            y_hat = np.dot(H, state)
            error = y[t] - y_hat
            Q = np.dot(H, np.dot(state_cov_prior, H)) + Ve
            K = np.dot(state_cov_prior, H) / Q
            state = state + K * error
            state_cov = state_cov_prior - np.outer(K, np.dot(H, state_cov_prior))

            intercepts[t] = state[0]
            hedge_ratios[t] = state[1]
            spread_errors[t] = error
            innovation_vars[t] = Q
            z_scores[t] = error / np.sqrt(Q) if Q > 0 else 0.0

        self.df['Kalman_Intercept'] = intercepts
        self.df['Hedge_Ratio'] = hedge_ratios
        self.df['Spread'] = spread_errors
        self.df['Kalman_Q'] = innovation_vars
        self.df['z_score'] = z_scores

        if self.compute_stationarity:
            adf_pvalues = np.full(n, np.nan)
            is_stat = np.full(n, False)

            for i in range(self.adf_window - 1, n):
                spread_slice = spread_errors[i - self.adf_window + 1 : i + 1]
                try:
                    adf_res = adfuller(spread_slice, maxlag=1)
                    p_val = adf_res[1]
                    adf_pvalues[i] = p_val
                    is_stat[i] = p_val < 0.10
                except Exception:
                    pass

            self.df['ADF_pvalue'] = adf_pvalues
            self.df['Is_Stationary'] = is_stat
        else:
            self.df['ADF_pvalue'] = np.nan
            self.df['Is_Stationary'] = False

        z_vals = self.df['z_score'].values
        hr_array = self.df['Hedge_Ratio'].values
        
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
