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

class PlateauOptimizer:
    """Locates the optimal plateau using the Top Performers Centroid method.

    Isolates the highest quantile of backtest results, identifies the dominant
    categorical regime, and selects the median continuous parameters to ensure
    structural robustness over fragile local peaks.

    Attributes:
        top_percentile (float): Proportion of top-performing parameter
            combinations to consider (e.g., 0.15 for the top 15%).
    """

    def __init__(self, top_percentile: float = 0.15) -> None:
        """Initializes the PlateauOptimizer.

        Args:
            top_percentile: Proportion of top results to include in the elite
                group. Defaults to 0.15 (evaluates the top 15% combinations).
        """
        self.top_percentile = top_percentile

    def select_best_plateau(
        self,
        df_results: pd.DataFrame,
        param_cols: List[str],
        metric_col: str = "recovery_factor",
    ) -> Dict[str, Any]:
        """Finds the robust parameter set at the center of the best plateau.

        Filters out NaNs and infinite values, extracts the top quantile based on
        `metric_col`, determines the modal values for categorical/discrete
        parameters, and picks the tested sample closest to the Euclidean centroid
        (medians) of the continuous parameters.

        Args:
            df_results: DataFrame containing backtest parameter combinations and
                their resulting performance metrics.
            param_cols: List of column names corresponding to strategy
                parameters.
            metric_col: The column name used to rank performance.
                Defaults to 'recovery_factor'.

        Returns:
            A dictionary containing the selected parameters alongside their
            'train_return' and the chosen evaluation metric. Returns an empty
            dictionary if `df_results` has no valid metric rows.
        """
        df = df_results.dropna(subset=[metric_col]).copy()

        # Remove any lingering infinite distortions
        df = df[df[metric_col] < float("inf")].reset_index(drop=True)
        if df.empty:
            return {}

        # 1. Isolate the Top Performers Quantile
        threshold = df[metric_col].quantile(1.0 - self.top_percentile)
        top_df = df[df[metric_col] >= threshold].copy()

        if len(top_df) < 3:
            best_idx = df[metric_col].idxmax()
            return self._extract_params(df, best_idx, param_cols, metric_col)

        # 2. Separate categorical (regime mapping) from continuous parameters
        cat_cols: List[str] = []
        num_cols: List[str] = []
        for col in param_cols:
            # Treats columns with strings or Nones as discrete categorical regimes
            if top_df[col].apply(
                lambda x: isinstance(x, str) or x is None or pd.isna(x)
            ).any():
                cat_cols.append(col)
            else:
                num_cols.append(col)

        # 3. Find the dominant categorical regime in the Top Performers
        dominant_regime: Dict[str, Any] = {}
        for col in cat_cols:
            # Gets the most frequent state (e.g., 'dynamic' winning over 15)
            dominant_regime[col] = top_df[col].value_counts(dropna=False).idxmax()

        # Filter the elite group to only those inside the dominant regime
        stable_zone = top_df.copy()
        for col, val in dominant_regime.items():
            if pd.isna(val):
                stable_zone = stable_zone[stable_zone[col].isna()]
            else:
                stable_zone = stable_zone[stable_zone[col] == val]

        # Fallback if the strict intersection is too narrow
        if stable_zone.empty:
            stable_zone = top_df

        # 4. Find the continuous centroid (Median) inside the stable zone
        centroid_idx = stable_zone[metric_col].idxmax()  # Fallback to localized peak
        if num_cols:
            medians = stable_zone[num_cols].median()
            # Selects the exact tested combination closest to the mathematical median
            distances = ((stable_zone[num_cols] - medians) ** 2).sum(axis=1)
            centroid_idx = distances.idxmin()

        return self._extract_params(df, centroid_idx, param_cols, metric_col)

    def _extract_params(
        self,
        df: pd.DataFrame,
        idx: int,
        param_cols: List[str],
        metric_col: str,
    ) -> Dict[str, Any]:
        """Extracts and sanitizes parameters and metrics for a specific row index.

        Args:
            df: DataFrame containing the parameter combinations and metrics.
            idx: Row index of the chosen parameter combination.
            param_cols: List of parameter column names to extract.
            metric_col: The primary metric column name.

        Returns:
            A dictionary with parameter keys mapped to their values (converting
            NaNs to None) along with 'train_return' and the specified metric.
        """
        raw_params = df.loc[idx, param_cols].to_dict()
        best_params: Dict[str, Any] = {}
        for k, v in raw_params.items():
            if pd.isna(v):
                best_params[k] = None
            else:
                best_params[k] = v

        best_params["train_return"] = df.loc[idx, "train_return"]
        best_params[metric_col] = df.loc[idx, metric_col]

        return best_params
