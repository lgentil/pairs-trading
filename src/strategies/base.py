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

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Executes the complete indicator, signal, position, and return pipeline.

        Returns:
            pd.DataFrame: Calculated strategy DataFrame.
        """
        pass


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
