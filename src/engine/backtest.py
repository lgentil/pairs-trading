from .analyzer import RollingSpreadAnalyzer
from .optimizer import PlateauOptimizer
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

class PairsBacktest:
    """Comprehensive backtesting and optimization engine for Pairs Trading and Statistical Arbitrage models.

    Supports arbitrary concrete strategy implementations conforming to the BasePairsStrategy interface.
    Provides multidimensional hyperparameter grid searches, in-sample optimization, continuous historical
    warmup for out-of-sample validation, correlation diagnostics (Pearson and Spearman Rank), automated
    slice-based visualization, and optional QuantStats performance reporting.

    Attributes:
        ticker_x (str): Ticker symbol for Asset X (independent variable).
        ticker_y (str): Ticker symbol for Asset Y (dependent variable).
        start_date (str): In-Sample starting date formatted as 'YYYY-MM-DD'.
        end_date_train (str): In-Sample ending date formatted as 'YYYY-MM-DD'.
        start_date_test (str): Out-of-Sample starting date formatted as 'YYYY-MM-DD'.
        end_date (str): Out-of-Sample ending date formatted as 'YYYY-MM-DD'.
        strategy_cls (Type[BasePairsStrategy]): Strategy class implementing BasePairsStrategy.
        strategy_param_grid (Dict[str, List[Any]]): Dictionary mapping strategy parameter names to search grids.
        show_heatmaps (bool): Flag determining whether optimization heatmaps are displayed by default.
        requires_ols_analyzer (bool): Automatically set to False for self-contained State-Space models (e.g., Kalman Filter).
        ols_windows (List[Union[int, str]]): List of rolling regression windows evaluated during optimization.
        strategy_name (str): Name of the Strategy.
        transaction_cost (float): Assumed slippage and commission per trade passed down to strategies.
        compounding (bool): Determines if returns are reinvested (True) or fixed size (False).
        df_train (Optional[pd.DataFrame]): Historical price data for the in-sample period.
        df_test (Optional[pd.DataFrame]): Historical price data for the out-of-sample period.
        best_train_df (Optional[pd.DataFrame]): Output DataFrame containing the optimized in-sample execution.
        best_test_df (Optional[pd.DataFrame]): Output DataFrame containing the out-of-sample validation run.
        best_params (Dict[str, Any]): Dictionary of the highest-performing hyperparameter combination found in-sample.
        optimization_df (Optional[pd.DataFrame]): Merged DataFrame containing in-sample and out-of-sample returns across all tested grids.
    """

    def __init__(
        self,
        ticker_x: str,
        ticker_y: str,
        start_date: str,
        end_date_train: str,
        start_date_test: str,
        end_date: str,
        strategy_cls: Type['BasePairsStrategy'],
        strategy_param_grid: Dict[str, List[Any]],
        strategy_name: str,
        ols_windows_to_test: Optional[List[int]] = None,
        show_heatmaps: bool = True,
        transaction_cost: float = 0.0,
        compounding: bool = False,
        target_metric: str = 'train_return'
        

    ) -> None:
        """Initializes the backtesting environment, sets operational flags, and formats parameter grids.

        Args:
            ticker_x (str): Ticker symbol for Asset X (independent variable).
            ticker_y (str): Ticker symbol for Asset Y (dependent variable).
            start_date (str): In-Sample starting date ('YYYY-MM-DD').
            end_date_train (str): In-Sample ending date ('YYYY-MM-DD').
            start_date_test (str): Out-of-Sample starting date ('YYYY-MM-DD').
            end_date (str): Out-of-Sample ending date ('YYYY-MM-DD').
            strategy_cls (Type[BasePairsStrategy]): Strategy class implementing BasePairsStrategy.
            strategy_param_grid (Dict[str, List[Any]]): Dictionary of hyperparameters to optimize.
            ols_windows_to_test (Optional[List[int]], optional): Lookback windows for OLS rolling regressions.
                Defaults to [60, 90, 120, 180, 252] if required.
            show_heatmaps (bool, optional): Whether to display heatmaps during plot generation. Defaults to True.
            transaction_cost (float, optional): Assumed slippage and commission per trade as a decimal. Defaults to 0.0.
            compounding (bool, optional): Compounding method for equity curve. Defaults to False for fixed sizing.
        """
        self.ticker_x: str = ticker_x
        self.ticker_y: str = ticker_y
        self.start_date: str = start_date
        self.end_date_train: str = end_date_train
        self.start_date_test: str = start_date_test
        self.end_date: str = end_date
        
        self.strategy_cls: Type['BasePairsStrategy'] = strategy_cls
        self.strategy_param_grid: Dict[str, List[Any]] = strategy_param_grid
        self.show_heatmaps: bool = show_heatmaps
        self.transaction_cost: float = float(transaction_cost)
        self.compounding: bool = bool(compounding)
        self.target_metric: str = target_metric
        self.strategy_name: str = strategy_name

        self.requires_ols_analyzer: bool = not strategy_cls.__name__ == 'KalmanZScoreStrategy'
        self.ols_windows: List[Union[int, str]] = (
            ols_windows_to_test if ols_windows_to_test is not None else [60, 90, 120, 180, 252]
        ) if self.requires_ols_analyzer else ['Kalman']

        self.df_train: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None
        self.best_train_df: Optional[pd.DataFrame] = None
        self.best_test_df: Optional[pd.DataFrame] = None

        self.best_params: Dict[str, Any] = {}
        self.results_train_records_: List[Dict[str, Any]] = []
        self.results_test_records_: List[Dict[str, Any]] = []
        self.optimization_df: Optional[pd.DataFrame] = None

    def _fetch_data(self, start: str, end: str, verbose: bool = True) -> pd.DataFrame:
        """Downloads historical price data and evaluates cointegration via the Engle-Granger test.

        Args:
            start (str): Start date formatted as 'YYYY-MM-DD'.
            end (str): End date formatted as 'YYYY-MM-DD'.
            verbose (bool, optional): Whether to output download dimensions and cointegration statistics. Defaults to True.

        Returns:
            pd.DataFrame: Cleaned DataFrame containing price series for Asset X and Asset Y.
        """
        if verbose:
            print("*" * 50)
            print(f" Downloading Historical Data for {self.ticker_x} and {self.ticker_y} via Yahoo Finance")

        df_x = yf.download(self.ticker_x, start=start, end=end, progress=False)
        if isinstance(df_x.columns, pd.MultiIndex):
            df_x = df_x.droplevel(1, axis=1)

        df_y = yf.download(self.ticker_y, start=start, end=end, progress=False)
        if isinstance(df_y.columns, pd.MultiIndex):
            df_y = df_y.droplevel(1, axis=1)

        col_x = "Adj Close" if "Adj Close" in df_x.columns else "Close"
        col_y = "Adj Close" if "Adj Close" in df_y.columns else "Close"

        df = pd.concat(
            [
                df_x[[col_x]].rename(columns={col_x: self.ticker_x}),
                df_y[[col_y]].rename(columns={col_y: self.ticker_y})
            ],
            axis=1
        ).dropna()


        return df

    def fetch_data(self) -> None:
        """Fetches and prepares both In-Sample (Train) and Out-of-Sample (Test) datasets."""
        print("--- Downloading and Testing Training Data (In-Sample) ---")
        self.df_train = self._fetch_data(self.start_date, self.end_date_train, verbose=True)

        print("--- Downloading and Testing Testing Data (Out-of-Sample) ---")
        self.df_test = self._fetch_data(self.start_date_test, self.end_date, verbose=True)

    def optimize_train(self) -> None:
        """Executes multidimensional grid search optimization on the In-Sample dataset."""
        param_names = list(self.strategy_param_grid.keys())
        param_values = list(self.strategy_param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        total_models = len(self.ols_windows) * len(param_combinations)
        print(f"--- Starting Optimization on Train Set ({total_models} models) ---")

        self.results_train_records_ = []
        
        # Frequency Filter: Minimum of 1 trade per semester
        total_days = (pd.to_datetime(self.df_train.index[-1]) - pd.to_datetime(self.df_train.index[0])).days
        min_trades_required = int(max(total_days / 365.25, 1.0) * 2) 

        if self.requires_ols_analyzer:
            for window in self.ols_windows:
                # CACHE: Run the heavy analyzer only ONCE per rolling window
                analyzer = RollingSpreadAnalyzer(self.df_train, self.ticker_x, self.ticker_y, window=window)
                df_analyzed = analyzer.run_analysis()

                for comb in param_combinations:
                    current_params = dict(zip(param_names, comb))
                    strategy = self.strategy_cls(
                        df_analyzed.copy(), 
                        self.ticker_x, 
                        self.ticker_y, 
                        transaction_cost=self.transaction_cost,
                        compounding=self.compounding,
                        **current_params
                    )
                    
                    perf = strategy.get_performance_metrics()
                    
                    # Zeroes the target metric if it does not reach statistical significance in trades
                    metric_value = perf.get(self.target_metric, 0.0)
                    if perf['trade_count'] < min_trades_required:
                        metric_value = 0.0
                    
                    record = {
                        'ols_window': window,
                        **current_params,
                        'train_return': perf['train_return'],
                        'recovery_factor': perf['recovery_factor'],
                        'sharpe_ratio': perf['sharpe_ratio'],
                        'profit_factor': perf['profit_factor'],
                        'trade_count': perf['trade_count'],
                        'target_optimization_metric': metric_value
                    }
                    self.results_train_records_.append(record)

        else:
            for comb in param_combinations:
                current_params = dict(zip(param_names, comb))
                strategy = self.strategy_cls(
                    self.df_train.copy(), 
                    self.ticker_x, 
                    self.ticker_y, 
                    transaction_cost=self.transaction_cost,
                    compounding=self.compounding,
                    **current_params
                )
                
                perf = strategy.get_performance_metrics()
                
                # Zeroes the target metric if it does not reach statistical significance in trades
                metric_value = perf.get(self.target_metric, 0.0)
                if perf['trade_count'] < min_trades_required:
                    metric_value = 0.0
                
                record = {
                    'ols_window': 'Kalman',
                    **current_params,
                    'train_return': perf['train_return'],
                    'recovery_factor': perf['recovery_factor'],
                    'sharpe_ratio': perf['sharpe_ratio'],
                    'profit_factor': perf['profit_factor'],
                    'trade_count': perf['trade_count'],
                    'target_optimization_metric': metric_value
                }
                self.results_train_records_.append(record)

        # Plateau Selection Process
        df_train_res = pd.DataFrame(self.results_train_records_)
        param_cols = ['ols_window'] + param_names
        
        plateau_selector = PlateauOptimizer(top_percentile=0.15)
        self.best_params = plateau_selector.select_best_plateau(
            df_results=df_train_res, 
            param_cols=param_cols, 
            metric_col='target_optimization_metric'
        )

        print(f"\n Best Hyperparameters Identified (Target: {self.target_metric}): {self.best_params}\n")

    def optimize_test(self) -> None:
        """Validates all parameter configurations out-of-sample using continuous historical warmup.

        Merges in-sample and out-of-sample return distributions into an internal optimization DataFrame.
        """
        print("--- Starting Out-of-Sample Validation (With Continuous Warmup) ---")
        param_names = list(self.strategy_param_grid.keys())
        param_values = list(self.strategy_param_grid.values())
        param_combinations = list(itertools.product(*param_values))

        self.results_test_records_ = []

        if self.requires_ols_analyzer:
            for window in self.ols_windows:
                overlap_size = window - 1
                df_test_extended = pd.concat([self.df_train.iloc[-overlap_size:], self.df_test])
                analyzer_test = RollingSpreadAnalyzer(df_test_extended, self.ticker_x, self.ticker_y, window=window)
                df_test_analyzed = analyzer_test.run_analysis()

                for comb in param_combinations:
                    params = dict(zip(param_names, comb))
                    strategy_test = self.strategy_cls(
                        df_test_analyzed.copy(), 
                        self.ticker_x, 
                        self.ticker_y, 
                        transaction_cost=self.transaction_cost,
                        compounding=self.compounding,
                        **params
                    )
                    
                    pure_oos_df = strategy_test.df.loc[self.df_test.index]
                    
                    if self.compounding:
                        oos_return = ((pure_oos_df['net_strategy_returns'].fillna(0.0) + 1.0).cumprod().iloc[-1] - 1.0) * 100.0
                    else:
                        oos_return = pure_oos_df['net_strategy_returns'].sum() * 100.0

                    test_record = {
                        'ols_window': window,
                        **params,
                        'test_return': oos_return
                    }
                    self.results_test_records_.append(test_record)
        else:
            df_full = pd.concat([self.df_train, self.df_test])
            for comb in param_combinations:
                params = dict(zip(param_names, comb))
                strategy_test = self.strategy_cls(
                    df_full.copy(), 
                    self.ticker_x, 
                    self.ticker_y, 
                    transaction_cost=self.transaction_cost,
                    compounding=self.compounding,
                    **params
                )
                
                pure_oos_df = strategy_test.df.loc[self.df_test.index]
                
                if self.compounding:
                    oos_return = ((pure_oos_df['net_strategy_returns'].fillna(0.0) + 1.0).cumprod().iloc[-1] - 1.0) * 100.0
                else:
                    oos_return = pure_oos_df['net_strategy_returns'].sum() * 100.0

                test_record = {
                    'ols_window': 'Kalman',
                    **params,
                    'test_return': oos_return
                }
                self.results_test_records_.append(test_record)

        df_train_res = pd.DataFrame(self.results_train_records_)
        df_test_res = pd.DataFrame(self.results_test_records_)
        self.optimization_df = pd.merge(
            df_train_res,
            df_test_res,
            on=['ols_window'] + list(self.strategy_param_grid.keys())
        )

    def plot_results(self, show_heatmaps: Optional[bool] = None, **filter_kwargs: Any) -> None:
        """Plots Heatmaps and interactive Scatter Plots with correlation metrics for a selected parameter slice.

        Args:
            show_heatmaps (Optional[bool], optional): Controls whether ANY plots are shown (heatmaps and scatter).
                Defaults to self.show_heatmaps if not provided. Set to False for clean batch processing.
            **filter_kwargs: Parameter slice filters (e.g., ma_window='dynamic', lookback_window=20, delta=1e-4).
        """
        # 1. Master Visual Switch: If False, abort the function immediately
        display_plots = show_heatmaps if show_heatmaps is not None else self.show_heatmaps
        if not display_plots:
            return

        if self.optimization_df is None:
            print("Error: Run optimize_train() and optimize_test() first.")
            return

        df_plot = self.optimization_df.copy()
        slice_title = ""

        # Filter by specifically requested kwargs
        if filter_kwargs:
            for param, val in filter_kwargs.items():
                if param in df_plot.columns:
                    df_plot = df_plot[df_plot[param] == val]
                    slice_title += f" ({param}: {val})"
        else:
            # For OLS models, lock the primary moving average window to its best parameter to avoid 3D overlap
            primary_window_key = next((k for k in ['lookback_window', 'ma_window'] if k in self.best_params), None)
            if primary_window_key:
                best_win = self.best_params[primary_window_key]
                df_plot = df_plot[df_plot[primary_window_key] == best_win]
                slice_title += f" ({primary_window_key}: {best_win})"

        if df_plot.empty:
            print("Error: No data available for the specified parameter slice.")
            return

        # ---------------------------------------------------------
        # BLOCK 1: HEATMAPS (Dynamic axes for OLS and State-Space models)
        # ---------------------------------------------------------
        y_col = None
        x_col = None
        other_params = []

        # Determine the best X and Y axes for the 2D Heatmap based on the model type
        if self.requires_ols_analyzer:
            y_col = 'ols_window'
            x_col = next((k for k in ['num_std', 'entry_z'] if k in self.strategy_param_grid), None)
            other_params = [
                k for k in self.strategy_param_grid.keys() 
                if k not in [x_col, 'lookback_window', 'ma_window'] and k not in filter_kwargs
            ]
        else:
            # For Kalman Filter or other non-OLS models, dynamically select the first two varying parameters for the grid axes
            varying_params = [
                k for k in self.strategy_param_grid.keys() 
                if len(self.strategy_param_grid[k]) > 1 and k not in filter_kwargs
            ]
            if len(varying_params) >= 2:
                y_col = varying_params[0]
                x_col = varying_params[1]
                other_params = [
                    k for k in self.strategy_param_grid.keys() 
                    if k not in [y_col, x_col] and k not in filter_kwargs
                ]
            else:
                print("Notice: Not enough varying parameters in the grid to generate a 2D heatmap for this model.")

        # If we successfully mapped two dimensions, generate the Heatmaps
        if y_col and x_col:
            df_heatmap = df_plot.copy()
            
            # Slice the dataframe by locking the remaining parameters to their optimized "best" values
            for p in other_params:
                if p in self.best_params:
                    param_val = self.best_params[p]
                    if pd.isna(param_val) or param_val is None:
                        df_heatmap = df_heatmap[df_heatmap[p].isna()]
                    else:
                        df_heatmap = df_heatmap[df_heatmap[p] == param_val]

            if not df_heatmap.empty:
                pivot_train = df_heatmap.pivot_table(index=y_col, columns=x_col, values='train_return', aggfunc='mean')
                pivot_test = df_heatmap.pivot_table(index=y_col, columns=x_col, values='test_return', aggfunc='mean')

                plt.figure(figsize=(14, 5))
                sns.heatmap(pivot_train, cmap='RdYlGn', annot=False)
                plt.title(f'Training Optimization: {y_col} vs {x_col}{slice_title} for the {self.strategy_name}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.show()

                plt.figure(figsize=(14, 5))
                sns.heatmap(pivot_test, cmap='RdYlGn', annot=False)
                plt.title(f'Out-of-Sample Validation: {y_col} vs {x_col}{slice_title} for the {self.strategy_name}')
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.show()

        # ---------------------------------------------------------
        # BLOCK 2: INTERACTIVE SCATTER PLOT (For all classes)
        # ---------------------------------------------------------
        pearson_corr = df_plot['train_return'].corr(df_plot['test_return'], method='pearson')
        spearman_corr = df_plot['train_return'].corr(df_plot['test_return'], method='spearman')

        if spearman_corr >= 0.7:
            status, color = "EXCELLENT", "green"
        elif spearman_corr >= 0.5:
            status, color = "GOOD", "blue"
        elif spearman_corr > 0.0:
            status, color = "WEAK", "orange"
        else:
            status, color = "OVERFITTED", "red"

        # Color the scatter plot dynamically: use OLS window for OLS models, and the first varying parameter for Kalman
        color_feature = 'ols_window' if self.requires_ols_analyzer else (y_col if y_col else None)
        if color_feature and color_feature in df_plot.columns:
            df_plot[str(color_feature)] = df_plot[color_feature].astype(str)
            color_arg = str(color_feature)
        else:
            color_arg = None

        hover_cols = [color_arg] + list(self.strategy_param_grid.keys()) if color_arg else list(self.strategy_param_grid.keys())

        fig = px.scatter(
            df_plot,
            x='train_return',
            y='test_return',
            color=color_arg,
            hover_data=hover_cols,
            title=f'In-Sample vs Out-of-Sample Performance Validation{slice_title} for the {self.strategy_name}',
            labels={'train_return': 'Training Return (%)', 'test_return': 'Out-of-Sample Test Return (%)'}
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.5)
        fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))


        annotation_text = (
            f"<b>Correlation Analysis</b><br>"
            f"Pearson (Linear): {pearson_corr:.4f}<br>"
            f"Spearman (Rank): {spearman_corr:.4f}<br>"
            f"Status: <span style='color:{color}'><b>{status}</b></span>"
        )

        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            align="left",
            font=dict(size=12, color="black"),
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=6
        )

        fig.show()

    def calculate_grid_correlation(self, group_by_param: Optional[str] = None) -> None:
        """Calculates and prints linear and rank correlation metrics segregated by parameter groups.

        Args:
            group_by_param (Optional[str], optional): Parameter name to group evaluations by. 
                Defaults to 'ma_window' for Bollinger or 'lookback_window' for Z-Score.
        """
        if self.optimization_df is None:
            print("Error: Run optimizations first.")
            return

        print("\n" + "=" * 65)
        print(f" IS vs OOS HYPERPARAMETER GRID CORRELATION - {self.strategy_name}")
        print("=" * 65)

        target_param = group_by_param
        if target_param is None:
            target_param = next((k for k in ['lookback_window', 'ma_window', 'delta'] if k in self.optimization_df.columns), None)

        if target_param and target_param in self.optimization_df.columns:
            for val in self.optimization_df[target_param].unique():
                df_sub = self.optimization_df[self.optimization_df[target_param] == val]
                pearson = df_sub['train_return'].corr(df_sub['test_return'], method='pearson')
                spearman = df_sub['train_return'].corr(df_sub['test_return'], method='spearman')

                status = (
                    "EXCELLENT" if spearman >= 0.7 
                    else "GOOD" if spearman >= 0.5 
                    else "WEAK" if spearman > 0 
                    else "OVERFITTED"
                )
                print(f" {target_param}: [{val}]")
                print(f"   -> Pearson (Linear): {pearson:.4f}")
                print(f"   -> Spearman (Rank):  {spearman:.4f}")
                print(f"   -> Status: {status}")
                print("-" * 60)
        else:
            pearson = self.optimization_df['train_return'].corr(self.optimization_df['test_return'], method='pearson')
            spearman = self.optimization_df['train_return'].corr(self.optimization_df['test_return'], method='spearman')
            print(f" Overall Pearson (Linear): {pearson:.4f}")
            print(f" Overall Spearman (Rank):  {spearman:.4f}")
            print("=" * 60 + "\n")

    def run_model_in_sample(self) -> Optional[pd.DataFrame]:
        """Executes the strategy across In-Sample price data using identified optimal parameters.

        Returns:
            Optional[pd.DataFrame]: Output DataFrame containing the optimized in-sample execution.
        """
        if not self.best_params:
            print("Error: Hyperparameters have not been calibrated.")
            return None

        params = {k: v for k, v in self.best_params.items() if k not in ['ols_window', 'train_return', 'test_return']}

        if self.requires_ols_analyzer:
            w = self.best_params['ols_window']
            analyzer = RollingSpreadAnalyzer(self.df_train, self.ticker_x, self.ticker_y, window=w)
            df_analyzed = analyzer.run_analysis()
            strategy = self.strategy_cls(
                df_analyzed, 
                self.ticker_x, 
                self.ticker_y, 
                transaction_cost=self.transaction_cost,
                compounding=self.compounding,
                **params
            )
        else:
            strategy = self.strategy_cls(
                self.df_train.copy(), 
                self.ticker_x, 
                self.ticker_y, 
                compute_stationarity=True, 
                transaction_cost=self.transaction_cost,
                compounding=self.compounding,
                **params
            )

        self.best_train_df = strategy.df
        return self.best_train_df

    def run_model_out_of_sample(self) -> Optional[pd.DataFrame]:
        """Executes the strategy across Out-of-Sample price data with continuous historical warmup.

        Returns:
            Optional[pd.DataFrame]: Output DataFrame containing the out-of-sample validation run.
        """
        if not self.best_params:
            print("Error: Hyperparameters have not been calibrated.")
            return None

        params = {k: v for k, v in self.best_params.items() if k not in ['ols_window', 'train_return', 'test_return']}

        if self.requires_ols_analyzer:
            w = self.best_params['ols_window']
            overlap_size = w - 1
            df_test_extended = pd.concat([self.df_train.iloc[-overlap_size:], self.df_test])
            analyzer = RollingSpreadAnalyzer(df_test_extended, self.ticker_x, self.ticker_y, window=w)
            df_analyzed = analyzer.run_analysis()
            strategy = self.strategy_cls(
                df_analyzed, 
                self.ticker_x, 
                self.ticker_y, 
                transaction_cost=self.transaction_cost,
                compounding=self.compounding,
                **params
            )
            self.best_test_df = strategy.df.loc[self.df_test.index]
        else:
            df_full = pd.concat([self.df_train, self.df_test])
            strategy = self.strategy_cls(
                df_full.copy(), 
                self.ticker_x, 
                self.ticker_y, 
                compute_stationarity=True, 
                transaction_cost=self.transaction_cost,
                compounding=self.compounding,
                **params
            )
            self.best_test_df = strategy.df.loc[self.df_test.index]

        return self.best_test_df

    def generate_quantstats_report(self, dataset_type: str = 'test', title: str = "Strategy Performance") -> None:
        """Generates a QuantStats HTML performance tear sheet benchmarked against Asset X.

        Args:
            dataset_type (str, optional): 'train' for In-Sample or 'test' for Out-of-Sample. Defaults to 'test'.
            title (str, optional): Display title of the report. Defaults to "Strategy Performance".
        """
        df = self.best_train_df if dataset_type.lower() == 'train' else self.best_test_df
        if df is None:
            print(f"Error: Data for '{dataset_type}' has not been populated.")
            return

        print(f"\n Generating QuantStats Report for {dataset_type.upper()} Series...")
        strategy_returns = df['net_strategy_returns'].fillna(0.0)
        strategy_returns.index = pd.to_datetime(strategy_returns.index)

        benchmark_returns = df[self.ticker_x].pct_change().fillna(0.0)
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

        qs.reports.full(
            returns=strategy_returns,
            benchmark=benchmark_returns,
            title=f"{title} ({dataset_type.upper()}) - Benchmark: {self.ticker_x}",
            match_dates=True
        )

    def execute_pipeline(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Executes the full automated workflow: fetch, train optimization, test validation, and plotting.

        Returns:
            Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]: In-Sample and Out-of-Sample DataFrames.
        """
        self.fetch_data()
        if self.df_train is not None and self.df_test is not None:
            self.optimize_train()
            self.optimize_test()
            self.plot_results()
            self.calculate_grid_correlation()
            return self.run_model_in_sample(), self.run_model_out_of_sample()
        return None, None
