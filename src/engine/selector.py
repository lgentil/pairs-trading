from .analyzer import RollingSpreadAnalyzer
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

class StatisticalAutoSelector:
    """Implements an Adaptive Auto-Selection Model based on structural spread statistics.

    Instead of relying on past performance metrics (e.g., Sharpe Ratio), this meta-model 
    evaluates the underlying time-series properties—specifically the Augmented Dickey-Fuller 
    (ADF) p-value and Ornstein-Uhlenbeck Half-Life—over a rolling window. It deterministically 
    routes capital to the most appropriate quantitative strategy for the forward step based on 
    the current market regime.

    Attributes:
        df_full (pd.DataFrame): The complete market dataset (In-Sample + Out-of-Sample).
        test_start_date (pd.Timestamp): The start date of the Out-of-Sample evaluation period.
        ticker_x (str): Ticker symbol for the independent asset.
        ticker_y (str): Ticker symbol for the dependent asset.
        strategies_config (dict): Mapping of strategy names to their respective Class and optimized parameters.
        lookback_days (int): Rolling evaluation window size for regime detection.
        step_days (int): Execution window size for the forward step.
        t_cost (float): Execution cost per trade leg for realistic transition penalties.
        precomputed_data (dict): Cached dataframes containing pre-processed spreads and hedge ratios.
    """
    
    def __init__(
        self,
        df_full: pd.DataFrame,
        test_start_date: str,
        ticker_x: str,
        ticker_y: str,
        strategies_config: dict,
        lookback_days: int = 90,
        step_days: int = 21,
        transaction_cost: float = 0.0004
    ) -> None:
        """Initializes the statistical regime selector and precomputes strategy-ready data."""
        self.df_full = df_full.copy()
        self.test_start_date = pd.to_datetime(test_start_date)
        self.ticker_x = ticker_x
        self.ticker_y = ticker_y
        self.strategies_config = strategies_config
        self.lookback_days = lookback_days
        self.step_days = step_days
        self.t_cost = float(transaction_cost)
        self.df_full.index = pd.to_datetime(self.df_full.index)
        self.precomputed_data = {}
        
        # Precompute the OLS Rolling Spread for the entire dataset
        for strat_name, (StratClass, params) in self.strategies_config.items():
            if "Kalman" not in StratClass.__name__:
                ols_w = params.get('ols_window', 90)
                analyzer = RollingSpreadAnalyzer(self.df_full.copy(), self.ticker_x, self.ticker_y, window=ols_w)
                self.precomputed_data[strat_name] = analyzer.run_analysis()
            else:
                self.precomputed_data[strat_name] = self.df_full.copy()

    def _evaluate_spread_regime(self, df_lookback: pd.DataFrame, lookback_start_idx: int, full_start_idx: int) -> str:
        """Evaluates structural integrity and compares recent performance."""
        try:
            analyzer = RollingSpreadAnalyzer(df_lookback.copy(), self.ticker_x, self.ticker_y, window=90)
            df_stats = analyzer.run_analysis()
            
            recent_adf = df_stats['ADF_pvalue'].dropna().tail(21).median()
            recent_hl = df_stats['Half_Life'].dropna().tail(21).median()
            
            if pd.isna(recent_adf):
                return 'Kalman_Filter'
                
            # If strictly non-stationary, fall back to Kalman
            if recent_adf > 0.10:
                return 'Kalman_Filter'
                
            # Spread is stationary (<= 0.10). We evaluate recent performance to choose between BB and Z-Score.
            metrics_to_ignore = [
                'train_return', 'test_return', 'target_optimization_metric',
                'recovery_factor', 'sharpe_ratio', 'profit_factor', 'trade_count', 'decision_score'
            ]
            
            recent_scores = {}
            for strat_name in ['Bollinger_Bands', 'Dynamic_ZScore']:
                if strat_name not in self.strategies_config:
                    continue
                    
                StratClass, params = self.strategies_config[strat_name]
                strat_params = {k: v for k, v in params.items() if k not in metrics_to_ignore}
                strat_params['compounding'] = False
                
                df_exec = self.precomputed_data[strat_name].iloc[lookback_start_idx:full_start_idx].copy()
                
                temp_strat = StratClass(
                    df=df_exec,
                    ticker_x=self.ticker_x,
                    ticker_y=self.ticker_y,
                    transaction_cost=self.t_cost,
                    **strat_params
                )
                
                perf = temp_strat.get_performance_metrics()
                # Prioritize Sharpe Ratio, fallback to train_return if Sharpe is zero
                score = perf.get('sharpe_ratio', 0.0)
                if score == 0.0:
                    score = perf.get('train_return', -100.0)
                
                recent_scores[strat_name] = score

            bb_score = recent_scores.get('Bollinger_Bands', -np.inf)
            zs_score = recent_scores.get('Dynamic_ZScore', -np.inf)
            
            # Use original statistical rules as a base, but override if performance clearly dictates otherwise
            if recent_adf <= 0.01 and recent_hl <= 20.0:
                # Highly stationary (Default BB territory). Override only if Z-score is vastly outperforming
                if zs_score > (bb_score + 1.0): 
                    return 'Dynamic_ZScore'
                return 'Bollinger_Bands'
            else:
                # Weakly stationary (Default ZS territory). Override if BB is outperforming
                if bb_score > zs_score:
                    return 'Bollinger_Bands'
                return 'Dynamic_ZScore'
                
        except Exception:
            return 'Kalman_Filter'

    def run(self) -> pd.DataFrame:
        """Executes the statistical walk-forward selection and generates the stitched curve."""
        oos_mask = self.df_full.index >= self.test_start_date
        oos_dates = self.df_full[oos_mask].index
        
        stitched_returns = []
        active_strategies = []
        executed_dates = []
        last_actual_position = 0.0
        
        total_oos_days = len(oos_dates)
        step_starts = range(0, total_oos_days, self.step_days)
        
        metrics_to_ignore = [
            'train_return', 'test_return', 'target_optimization_metric',
            'recovery_factor', 'sharpe_ratio', 'profit_factor', 'trade_count', 'decision_score'
        ]
        
        for i in step_starts:
            current_start_date = oos_dates[i]
            end_idx = min(i + self.step_days, total_oos_days) - 1
            current_end_date = oos_dates[end_idx]
            
            full_start_idx = self.df_full.index.get_loc(current_start_date)
            lookback_start_idx = max(0, full_start_idx - self.lookback_days)
            
            df_lookback = self.df_full.iloc[lookback_start_idx:full_start_idx].copy()
            
            # Pass indices to evaluate_spread_regime to compute tie-breakers without re-indexing heavy DataFrames
            best_strategy_name = self._evaluate_spread_regime(df_lookback, lookback_start_idx, full_start_idx)
            
            StratClass, params = self.strategies_config[best_strategy_name]
            strat_params = {k: v for k, v in params.items() if k not in metrics_to_ignore}
            strat_params['compounding'] = False
            
            df_exec = self.precomputed_data[best_strategy_name].iloc[lookback_start_idx:]
            
            forward_strat = StratClass(
                df=df_exec,
                ticker_x=self.ticker_x,
                ticker_y=self.ticker_y,
                transaction_cost=self.t_cost,
                **strat_params
            )
            
            step_results = forward_strat.df.loc[current_start_date:current_end_date]
            step_returns = step_results['net_strategy_returns'].fillna(0.0).values
            step_positions = step_results['positions'].fillna(0.0).values
            
            assumed_prev_pos = forward_strat.df['positions'].shift(1).loc[current_start_date]
            if pd.isna(assumed_prev_pos):
                assumed_prev_pos = 0.0
                
            current_pos = step_positions[0]
            cost_charged = abs(current_pos - assumed_prev_pos) * self.t_cost
            true_transition_cost = abs(current_pos - last_actual_position) * self.t_cost
            
            step_returns[0] = step_returns[0] + cost_charged - true_transition_cost
            last_actual_position = step_positions[-1]
            
            stitched_returns.extend(step_returns)
            active_strategies.extend([best_strategy_name] * len(step_returns))
            executed_dates.extend(self.df_full.index[full_start_idx:full_start_idx + len(step_returns)])
            
        wfo_results = pd.DataFrame({
            'selected_strategy': active_strategies,
            'net_strategy_returns': stitched_returns
        }, index=executed_dates)
        
        wfo_results['cumulative_returns'] = (wfo_results['net_strategy_returns'] + 1.0).cumprod()
        return wfo_results
