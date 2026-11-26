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

class BasePairsStrategy(ABC):
    """Abstract Base Class for modular Pairs Trading and Statistical Arbitrage strategies.

    Attributes:
        df (pd.DataFrame): DataFrame containing price data, hedge ratios, and spreads.
        ticker_x (str): Ticker symbol for Asset X.
        ticker_y (str): Ticker symbol for Asset Y.
        params (Dict[str, Any]): Dictionary of strategy-specific parameters.
    """

    def __init__(self, df: pd.DataFrame, ticker_x: str, ticker_y: str, **params: Any) -> None:
        """Initializes the base pairs strategy instance.

        Args:
            df (pd.DataFrame): Market dataset containing spread metrics.
            ticker_x (str): Ticker symbol for Asset X.
            ticker_y (str): Ticker symbol for Asset Y.
            **params: Dynamic keyword parameters.
        """
        self.df: pd.DataFrame = df.copy()
        self.ticker_x: str = ticker_x
        self.ticker_y: str = ticker_y
        self.params: Dict[str, Any] = params

        self.trading_mode: str = params.get('trading_mode', 'swing').lower()
        self.trade_only_stationary: bool = bool(params.get('trade_only_stationary', False))

        # Trading schedule configuration (for day_trade)
        entry_start_str = params.get('entry_start_time', '10:15')
        entry_end_str = params.get('entry_end_time', '16:30')
        close_str = params.get('close_time', '16:45')

        self.entry_start_time = pd.to_datetime(entry_start_str, format='%H:%M').time() if isinstance(entry_start_str, str) else entry_start_str
        self.entry_end_time = pd.to_datetime(entry_end_str, format='%H:%M').time() if isinstance(entry_end_str, str) else entry_end_str
        self.close_time = pd.to_datetime(close_str, format='%H:%M').time() if isinstance(close_str, str) else close_str

    def can_open_position(self, current_time: Any, is_stationary: Optional[bool] = None) -> bool:
        """Determines if a new position can be entered at the given timestamp.

        Args:
            current_time (Any): Current bar timestamp.
            is_stationary (Optional[bool]): Stationarity confirmation flag at bar close (ADF p-value < 0.10).

        Returns:
            bool: True if position opening is permitted, False otherwise.
        """
        # Strictly require confirmed stationarity at bar close if trade_only_stationary is enabled
        if self.trade_only_stationary:
            if is_stationary is None or not is_stationary:
                return False

        if self.trading_mode != 'day_trade':
            return True
        t = current_time.time() if hasattr(current_time, 'time') else current_time
        return self.entry_start_time <= t <= self.entry_end_time

    def must_close_position(self, current_time: Any) -> bool:
        """Determines if any active position must be forcibly closed for day trading."""
        if self.trading_mode != 'day_trade':
            return False
        t = current_time.time() if hasattr(current_time, 'time') else current_time
        return t >= self.close_time

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Executes the complete indicator, signal, position, and return pipeline.

        Returns:
            pd.DataFrame: Calculated strategy DataFrame.
        """
        pass

    def compute_strategy_returns(self) -> pd.DataFrame:
        """Calculates strategy returns using the next bar's Open price for trade execution.

        When a trading signal triggers at the Close of bar t:
        - Execution occurs at the Open of bar t+1 (market entry/exit at the open).
        - If Open prices are unavailable, gracefully falls back to Close prices.

        Returns:
            pd.DataFrame: Strategy DataFrame populated with return series.
        """
        close_y = self.df[self.ticker_y]
        close_x = self.df[self.ticker_x]

        col_open_y = f"{self.ticker_y}_open" if f"{self.ticker_y}_open" in self.df.columns else None
        col_open_x = f"{self.ticker_x}_open" if f"{self.ticker_x}_open" in self.df.columns else None

        open_y = self.df[col_open_y] if col_open_y else close_y.shift(1)
        open_x = self.df[col_open_x] if col_open_x else close_x.shift(1)

        pos = self.df['positions'].fillna(0.0)
        hr = self.df['trade_hr'].fillna(0.0)

        pos_lag1 = pos.shift(1).fillna(0.0)
        pos_lag2 = pos.shift(2).fillna(0.0)
        hr_lag1 = hr.shift(1).fillna(0.0)
        hr_lag2 = hr.shift(2).fillna(0.0)

        # Transition masks
        is_entry = (pos_lag2 == 0.0) & (pos_lag1 != 0.0)
        is_hold = (pos_lag2 != 0.0) & (pos_lag1 != 0.0) & (pos_lag1 == pos_lag2)
        is_exit = (pos_lag2 != 0.0) & (pos_lag1 == 0.0)
        is_reversal = (pos_lag2 != 0.0) & (pos_lag1 != 0.0) & (pos_lag1 != pos_lag2)

        # 1. ENTRY (Open[t] -> Close[t])
        dy_entry = close_y - open_y
        dx_entry = close_x - open_x
        pnl_entry = pos_lag1 * (dy_entry - (hr_lag1 * dx_entry))
        cap_entry = open_y.abs() + (hr_lag1.abs() * open_x.abs())

        # 2. HOLD / MAINTENANCE (Close[t-1] -> Close[t])
        dy_hold = close_y - close_y.shift(1)
        dx_hold = close_x - close_x.shift(1)
        pnl_hold = pos_lag1 * (dy_hold - (hr_lag1 * dx_hold))
        cap_hold = close_y.shift(1).abs() + (hr_lag1.abs() * close_x.shift(1).abs())

        # 3. EXIT (Close[t-1] -> Open[t])
        dy_exit = open_y - close_y.shift(1)
        dx_exit = open_x - close_x.shift(1)
        pnl_exit = pos_lag2 * (dy_exit - (hr_lag2 * dx_exit))
        cap_exit = close_y.shift(1).abs() + (hr_lag2.abs() * close_x.shift(1).abs())

        basket_pnl = np.where(
            is_entry, pnl_entry,
            np.where(
                is_hold, pnl_hold,
                np.where(
                    is_exit, pnl_exit,
                    np.where(is_reversal, pnl_entry + pnl_exit, 0.0)
                )
            )
        )

        allocated_capital = np.where(
            is_entry, cap_entry,
            np.where(
                is_hold, cap_hold,
                np.where(
                    is_exit, cap_exit,
                    np.where(is_reversal, np.maximum(cap_entry, cap_exit), 1.0)
                )
            )
        )
        allocated_capital = np.where(allocated_capital == 0, 1.0, allocated_capital)

        self.df['percentage_change'] = basket_pnl / allocated_capital
        self.df['strategy_returns'] = self.df['percentage_change']

        position_diff = pos.diff().fillna(0.0).abs()
        t_cost = getattr(self, 'transaction_cost', 0.0)
        self.df['transaction_costs'] = position_diff * t_cost
        self.df['net_strategy_returns'] = self.df['strategy_returns'].fillna(0.0) - self.df['transaction_costs']

        compounding = getattr(self, 'compounding', False)
        if compounding:
            self.df['cumulative_returns'] = (self.df['net_strategy_returns'] + 1.0).cumprod()
        else:
            self.df['cumulative_returns'] = self.df['net_strategy_returns'].cumsum() + 1.0

        return self.df

    def get_trade_count(self) -> int:
        """Calculates the total number of executed trades.

        Returns:
            int: The number of completed trades based on position changes.
        """
        if 'positions' not in self.df.columns:
            return 0
            
        # Each position change (e.g., 0 to 1, or 1 to 0) counts as half a trade.
        # Direct reversals (1 to -1) sum to 2. Dividing the absolute sum by 2 gives total round-trips.
        position_diff = self.df['positions'].diff().fillna(0.0).abs()
        return int(position_diff.sum() / 2.0)

    
    def get_performance_metrics(self) -> dict:
        """Calculates a comprehensive set of performance and risk metrics.

        Computes core strategy evaluation metrics including net profit, trade 
        frequency, risk-adjusted returns (Sharpe Ratio), and custom penalty-based 
        scoring models (Decision Score) to evaluate robustness.

        Returns:
            dict: Dictionary containing 'train_return', 'recovery_factor', 
                  'sharpe_ratio', 'profit_factor', 'trade_count', and 
                  'decision_score' (Sharpe Ratio penalized by Max Drawdown).
        """
        metrics = {
            'train_return': 0.0,
            'recovery_factor': 0.0,
            'sharpe_ratio': 0.0,
            'profit_factor': 0.0,
            'trade_count': 0,
            'decision_score': 0.0
        }
        
        if 'cumulative_returns' not in self.df.columns or self.df['cumulative_returns'].empty:
            return metrics
            
        cum_returns = self.df['cumulative_returns'].dropna()
        if cum_returns.empty:
            return metrics
            
        # 1. Total Return (Net Profit)
        net_profit = cum_returns.iloc[-1] - 1.0
        metrics['train_return'] = net_profit * 100.0
        
        # 2. Trade Count
        if 'positions' in self.df.columns:
            position_diff = self.df['positions'].diff().fillna(0.0).abs()
            metrics['trade_count'] = int(position_diff.sum() / 2.0)
            
        # 3. Recovery Factor and Max Drawdown Calculation
        running_max = cum_returns.cummax()
        drawdown = (running_max - cum_returns) / running_max
        
        # 0.5% floor to prevent division by zero and infinite spikes in recovery factor
        capped_max_dd = max(drawdown.max(), 0.005) 
        metrics['recovery_factor'] = net_profit / capped_max_dd
        
        # Actual Maximum Drawdown for the Decision Score
        actual_max_dd = drawdown.max()
        
        # 4. Sharpe Ratio (Annualized)
        sharpe = 0.0
        if 'net_strategy_returns' in self.df.columns:
            daily_returns = self.df['net_strategy_returns'].dropna()
            volatility = daily_returns.std()
            if volatility > 0:
                sharpe = (daily_returns.mean() / volatility) * math.sqrt(252)
                metrics['sharpe_ratio'] = sharpe
                
            # 5. Profit Factor
            gains = daily_returns[daily_returns > 0].sum()
            losses = abs(daily_returns[daily_returns < 0].sum())
            metrics['profit_factor'] = (gains / losses) if losses > 0 else float('inf')
            
        # 6. Custom Decision Score (Risk-Penalized Sharpe Ratio)
        # Formula: Sharpe Ratio * (1 - Max Drawdown)
        metrics['decision_score'] = sharpe * (1.0 - actual_max_dd)
            
        return metrics


    def get_total_return(self) -> float:
        """Retrieves the total accumulated percentage return of the strategy.

        Returns:
            float: Total cumulative return expressed as a percentage (e.g., 15.5 for 15.5%).
        """
        if 'cumulative_returns' not in self.df.columns or self.df['cumulative_returns'].empty:
            return 0.0
        last_valid_return = self.df['cumulative_returns'].dropna().iloc[-1]
        return (last_valid_return - 1.0) * 100.0
