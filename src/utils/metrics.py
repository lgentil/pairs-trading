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



def evaluate_stationarity_regimes(
    strategy_df: pd.DataFrame, 
    strategy_name: str, 
    compounding: bool = True
) -> None:
    """Evaluates strategy performance segregated across Stationary and Non-Stationary market regimes.

    Calculates active exposure days, cumulative returns, annualized returns (CAGR or Simple), 
    annualized volatility, and Sharpe ratios for periods identified by the rolling 
    Augmented Dickey-Fuller (ADF) test (p-value threshold of 0.10).

    Args:
        strategy_df (pd.DataFrame): Output DataFrame produced by a concrete strategy instance.
        strategy_name (str): Name of the strategy
        compounding (bool): If True, calculates compounded (reinvested) returns. If False, uses simple additive returns.
    """
    df = strategy_df.copy()

    stat_returns = df.loc[df['Is_Stationary'] == True, 'net_strategy_returns'].fillna(0.0)
    non_stat_returns = df.loc[df['Is_Stationary'] == False, 'net_strategy_returns'].fillna(0.0)

    stat_active_days = len(stat_returns[stat_returns != 0])
    non_stat_active_days = len(non_stat_returns[non_stat_returns != 0])

    stat_total_days = len(stat_returns)
    non_stat_total_days = len(non_stat_returns)

    # 1. Calculation Logic based on Compounding Mode
    if compounding:
        calc_mode_str = "COMPOUNDING (Reinvested)"
        
        stat_cum_return = (stat_returns + 1.0).prod() - 1.0
        non_stat_cum_return = (non_stat_returns + 1.0).prod() - 1.0

        stat_ann_return = ((1.0 + stat_cum_return) ** (252 / stat_total_days) - 1.0) if (stat_total_days > 0 and stat_cum_return > -1.0) else 0.0
        non_stat_ann_return = ((1.0 + non_stat_cum_return) ** (252 / non_stat_total_days) - 1.0) if (non_stat_total_days > 0 and non_stat_cum_return > -1.0) else 0.0
        
        df['stat_equity'] = (stat_returns + 1.0).cumprod()
        df['non_stat_equity'] = (non_stat_returns + 1.0).cumprod()
    else:
        calc_mode_str = "SIMPLE (Fixed Size / Additive)"
        
        stat_cum_return = stat_returns.sum()
        non_stat_cum_return = non_stat_returns.sum()
        
        stat_ann_return = (stat_cum_return / stat_total_days) * 252 if stat_total_days > 0 else 0.0
        non_stat_ann_return = (non_stat_cum_return / non_stat_total_days) * 252 if non_stat_total_days > 0 else 0.0
        
        df['stat_equity'] = stat_returns.cumsum() + 1.0
        df['non_stat_equity'] = non_stat_returns.cumsum() + 1.0

    # 2. Volatility and Sharpe Ratio
    stat_vol = stat_returns.std() * np.sqrt(252)
    non_stat_vol = non_stat_returns.std() * np.sqrt(252)

    stat_sharpe = (stat_returns.mean() * 252) / stat_vol if stat_vol > 0 else 0.0
    non_stat_sharpe = (non_stat_returns.mean() * 252) / non_stat_vol if non_stat_vol > 0 else 0.0

    # 3. Print Results
    print(f"\n=== Performance by Stationarity Regime [{calc_mode_str}] ===")
    print("[STATIONARY REGIME (ADF p-value < 0.10)]")
    print(f" Total Regime Days:   {stat_total_days}")
    print(f" Active Trading Days: {stat_active_days}")
    print(f" Cumulative Return:   {stat_cum_return * 100:.2f}%")
    print(f" Annualized Return:   {stat_ann_return * 100:.2f}%")
    print(f" Annualized Vol:      {stat_vol * 100:.2f}%")
    print(f" Sharpe Ratio:        {stat_sharpe:.2f}\n")

    print("[NON-STATIONARY REGIME (ADF p-value >= 0.10)]")
    print(f" Total Regime Days:   {non_stat_total_days}")
    print(f" Active Trading Days: {non_stat_active_days}")
    print(f" Cumulative Return:   {non_stat_cum_return * 100:.2f}%")
    print(f" Annualized Return:   {non_stat_ann_return * 100:.2f}%")
    print(f" Annualized Vol:      {non_stat_vol * 100:.2f}%")
    print(f" Sharpe Ratio:        {non_stat_sharpe:.2f}")

    # 4. Plotting
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df['stat_equity'], label='Stationary Days Regime', color='green')
    plt.plot(df.index, df['non_stat_equity'], label='Non-Stationary Days Regime', color='red')
    plt.title(f'Equity Curve Segregation by Market Regime [{calc_mode_str}] - {strategy_name}')
    plt.ylabel('Cumulative Growth Factor')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def apply_volatility_position_sizing(
    strategy_df: pd.DataFrame,
    vol_lookback: int = 20,
    target_vol_annualized: float = 0.30,
    mode: str = 'inverse',
    min_multiplier: float = 1.00,
    max_multiplier: float = 4.00,
    fixed_leverage: float = 4.00, 
    budget_neutral: bool = False,
    compounding: bool = False,
    title: str = 'Strategy'
) -> pd.DataFrame:
    """Applies dynamic position sizing based on realized rolling volatility.

    Calculates the rolling volatility of active trading periods and dynamically 
    adjusts the strategy's leverage (position size) to match a specified target 
    annualized volatility. Includes options for inverse or proportional scaling, 
    leverage caps, budget-neutral normalization, and reinvestment (compounding) 
    preferences for performance metrics.

    Args:
        strategy_df (pd.DataFrame): DataFrame containing 'positions' and 'net_strategy_returns'.
        vol_lookback (int, optional): Rolling window size (in days) to calculate realized volatility. Defaults to 20.
        target_vol_annualized (float, optional): The target annualized volatility to scale against. Defaults to 0.30 (30%).
        mode (str, optional): Sizing logic. 'inverse' scales up exposure when volatility is low; 'proportional' scales up when volatility is high. Defaults to 'inverse'.
        min_multiplier (float, optional): Floor boundary for the position sizing multiplier. Defaults to 1.00.
        max_multiplier (float, optional): Ceiling boundary for the position sizing multiplier. Defaults to 4.00.
        fixed_leverage (float, optional): Static leverage factor used solely for benchmark comparison. Defaults to 4.00.
        budget_neutral (bool, optional): If True, normalizes the active multipliers so their historical average equals 1.0. Defaults to False.
        compounding (bool, optional): If True, calculates cumulative returns with reinvestment. If False, computes simple additive returns. Defaults to False.
        title (str, optional): Strategy identifier used in the plot title. Defaults to 'Strategy'.

    Returns:
        pd.DataFrame: Augmented DataFrame containing rolling volatility, dynamic multipliers, and scaled strategy returns.
    """
    
    df = strategy_df.copy()

    # 1. Active Mask and Realized Volatility Calculation
    active_mask = (df['positions'].fillna(0.0) != 0)
    
    # Safely handle the column to compute standard deviation
    if 'percentage_change' in df.columns:
        active_returns = df.loc[active_mask, 'percentage_change']
    else:
        active_returns = df.loc[active_mask, 'net_strategy_returns']
        
    active_vol = active_returns.rolling(window=vol_lookback, min_periods=2).std(ddof=1) * np.sqrt(252)

    realized_vol = pd.Series(np.nan, index=df.index)
    realized_vol.loc[active_mask] = active_vol
    realized_vol = realized_vol.ffill().fillna(target_vol_annualized)

    # 2. Dynamic Volatility Sizing Logic
    if mode == 'proportional':
        raw_multiplier = np.maximum(realized_vol, 0.005) / target_vol_annualized
    else:
        raw_multiplier = target_vol_annualized / np.maximum(realized_vol, 0.005)

    clipped_multiplier = np.clip(raw_multiplier, min_multiplier, max_multiplier)

    if budget_neutral:
        avg_weight = clipped_multiplier[active_mask].mean()
        df['vol_multiplier'] = 1.0
        if avg_weight > 0:
            df.loc[active_mask, 'vol_multiplier'] = clipped_multiplier[active_mask] / avg_weight
    else:
        df['vol_multiplier'] = clipped_multiplier

    weight_lagged = df['vol_multiplier'].shift(1).fillna(1.0)

    # 3. Calculation of Comparative Return Curves
    df['vol_strategy_returns'] = df['net_strategy_returns'] * weight_lagged
    df['fixed_strategy_returns'] = df['net_strategy_returns'] * fixed_leverage

    # Conditional Compounding Application
    if compounding:
        base_cum = (df['net_strategy_returns'].fillna(0.0) + 1.0).cumprod()
        fixed_cum = (df['fixed_strategy_returns'].fillna(0.0) + 1.0).cumprod()
        vol_cum = (df['vol_strategy_returns'].fillna(0.0) + 1.0).cumprod()
    else:
        base_cum = df['net_strategy_returns'].fillna(0.0).cumsum() + 1.0
        fixed_cum = df['fixed_strategy_returns'].fillna(0.0).cumsum() + 1.0
        vol_cum = df['vol_strategy_returns'].fillna(0.0).cumsum() + 1.0

    # 4. Visualization and Reporting
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, base_cum, label='Baseline Net (1.0x)', color='gray', linestyle='--')
    plt.plot(df.index, fixed_cum, label=f'Fixed Leverage ({fixed_leverage}x)', color='orange', linestyle='-.')
    plt.plot(df.index, vol_cum, label=f'Adaptive Sizing (Vol Target: {target_vol_annualized*100:.1f}%)', color='blue', linewidth=1.8)
    
    compounding_label = "Compounded" if compounding else "Simple Additive (No Reinvestment)"
    plt.title(f'Performance Comparison [{compounding_label}]: Baseline vs. Fixed vs. Adaptive for {title}')
    plt.ylabel('Cumulative Growth Factor')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.show()

    print("=" * 65)
    print(f" 📊 LEVERAGE: DYNAMIC vs FIXED | MODE: {compounding_label.upper()}")
    print("=" * 65)
    
    for name, col in [('Baseline (1.0x)', 'net_strategy_returns'), 
                      (f'Fixed ({fixed_leverage}x)', 'fixed_strategy_returns'), 
                      ('Adaptive (Vol)', 'vol_strategy_returns')]:
        
        # Calculate Returns Based on Compounding Flag
        if compounding:
            cum_ret = ((df[col].fillna(0.0) + 1.0).cumprod().iloc[-1] - 1.0) * 100.0
        else:
            cum_ret = df[col].fillna(0.0).sum() * 100.0
            
        vol = df[col].std(ddof=1) * np.sqrt(252) * 100.0
        sharpe = (df[col].mean() * 252) / (df[col].std(ddof=1) * np.sqrt(252)) if vol > 0 else 0.0
        print(f" {name:<17} Net: {cum_ret:>7.2f}% | Vol: {vol:>6.2f}% | Sharpe: {sharpe:>5.2f}")
        
    print("=" * 65 + "\n")

    return df

def compare_strategy_statistics(
    dfs: list[pd.DataFrame],
    labels: list[str],
    compounding: bool = False,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Print and return a comparison of three trading strategies.

    Each input DataFrame must have a DatetimeIndex and these columns:
        - positions
        - strategy_returns
        - net_strategy_returns

    Returns must be decimal values: 0.01 means 1%.

    The function calculates portfolio-level metrics from the existing
    gross and net daily return series. It does not charge costs again.

    Trade-level returns follow the boundaries used by the original
    print_trade_statistics function: the entry row is excluded, and
    the exit row is included. Consequently, trade-level P&L may not
    reconcile exactly with portfolio-level P&L when costs are charged
    on entry or reversal rows.

    Args:
        dfs: Exactly three strategy DataFrames.
        labels: Exactly three distinct strategy names.
        compounding: Whether to compound returns when building the
            portfolio equity curve. False uses a fixed-size, additive
            equity curve.
        annualization_factor: Trading periods per year. The default
            of 252 assumes daily trading data.
        risk_free_rate: Annual risk-free rate as a decimal. The
            default is 0.0.

    Returns:
        A tuple containing:
            summary_df: Metrics as rows and strategies as columns.
            trade_logs: A dictionary mapping each label to its
                individual trade log.
    """
    if len(dfs) != 3 or len(labels) != 3:
        raise ValueError("Provide exactly three DataFrames and three labels.")

    if len(set(labels)) != 3:
        raise ValueError("Strategy labels must be unique.")

    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive.")

    if risk_free_rate <= -1.0:
        raise ValueError("risk_free_rate must be greater than -1.")

    required_columns = {
        "positions",
        "strategy_returns",
        "net_strategy_returns",
    }

    metric_names = [
        "Start date",
        "End date",
        "Trading days",
        "Period (months)",
        "Period (years)",
        "Gross return (%)",
        "Cost drag (%)",
        "Net return (%)",
        "Net CAGR (%)",
        "Maximum drawdown (%)",
        "Annual volatility (%)",
        "Annual Sharpe ratio",
        "Total trades",
        "Long trades",
        "Short trades",
        "Winning trades",
        "Losing trades",
        "Breakeven trades",
        "Win rate (%)",
        "Average winning trade (%)",
        "Average losing trade (%)",
        "Payoff ratio",
        "Win/loss count ratio",
        "Average trade duration (days)",
        "Minimum trade duration (days)",
        "Maximum trade duration (days)",
        "Trades per month",
    ]

    summary_records = {}
    trade_logs = {}

    for strategy_df, label in zip(dfs, labels):
        if not isinstance(strategy_df, pd.DataFrame):
            raise TypeError(f"{label}: input must be a pandas DataFrame.")

        if strategy_df.empty:
            raise ValueError(f"{label}: DataFrame is empty.")

        missing_columns = required_columns - set(strategy_df.columns)

        if missing_columns:
            raise ValueError(
                f"{label}: missing columns: {sorted(missing_columns)}"
            )

        df = strategy_df.copy()
        df.index = pd.to_datetime(df.index, errors="raise")

        if df.index.hasnans:
            raise ValueError(f"{label}: the index contains invalid dates.")

        if df.index.has_duplicates:
            raise ValueError(f"{label}: the index contains duplicate dates.")

        df = df.sort_index()

        for column in required_columns:
            df[column] = pd.to_numeric(df[column], errors="raise")

            if np.isinf(df[column].to_numpy(dtype=float)).any():
                raise ValueError(
                    f"{label}: column '{column}' contains infinite values."
                )

        positions = df["positions"].fillna(0.0)
        gross_returns = df["strategy_returns"].fillna(0.0)
        net_returns = df["net_strategy_returns"].fillna(0.0)

        if compounding and (
            (gross_returns <= -1.0).any()
            or (net_returns <= -1.0).any()
        ):
            raise ValueError(
                f"{label}: daily returns at or below -100% "
                "cannot be used for compounded equity."
            )

        dates = pd.DatetimeIndex(df.index)
        calendar_days = (dates[-1] - dates[0]).days
        years = calendar_days / 365.25
        months = calendar_days / 30.4375

        # Include initial capital in both equity curves.
        if compounding:
            gross_equity = np.concatenate(
                (
                    [1.0],
                    np.cumprod(1.0 + gross_returns.to_numpy()),
                )
            )
            net_equity = np.concatenate(
                (
                    [1.0],
                    np.cumprod(1.0 + net_returns.to_numpy()),
                )
            )
        else:
            gross_equity = np.concatenate(
                (
                    [1.0],
                    1.0 + np.cumsum(gross_returns.to_numpy()),
                )
            )
            net_equity = np.concatenate(
                (
                    [1.0],
                    1.0 + np.cumsum(net_returns.to_numpy()),
                )
            )

        gross_return_pct = (gross_equity[-1] - 1.0) * 100.0
        net_return_pct = (net_equity[-1] - 1.0) * 100.0

        # Cost drag is a performance difference, not a second cost charge.
        cost_drag_pp = gross_return_pct - net_return_pct

        if np.any(net_equity <= 0.0):
            cagr_pct = np.nan
            max_drawdown_pct = np.nan
        else:
            equity_peaks = np.maximum.accumulate(net_equity)
            drawdowns = net_equity / equity_peaks - 1.0
            max_drawdown_pct = float(drawdowns.min() * 100.0)

            cagr_pct = (
                (net_equity[-1] ** (1.0 / years) - 1.0) * 100.0
                if years > 0.0
                else np.nan
            )

        if len(net_returns) > 1:
            daily_std = float(net_returns.std(ddof=1))
            annual_volatility_pct = (
                daily_std * np.sqrt(annualization_factor) * 100.0
            )
            daily_risk_free_rate = (
                (1.0 + risk_free_rate)
                ** (1.0 / annualization_factor)
                - 1.0
            )

            sharpe_ratio = (
                (
                    float(net_returns.mean()) - daily_risk_free_rate
                )
                / daily_std
                * np.sqrt(annualization_factor)
                if daily_std > 0.0
                else np.nan
            )
        else:
            annual_volatility_pct = np.nan
            sharpe_ratio = np.nan

        # Identify trade boundaries using the original function's rules.
        position_values = positions.to_numpy()
        return_values = net_returns.to_numpy()
        trade_records = []

        in_trade = False
        entry_idx = None
        current_side = 0.0

        for i, current_position in enumerate(position_values):
            if in_trade and (
                current_position != current_side
                or current_position == 0.0
            ):
                trade_returns = return_values[entry_idx + 1 : i + 1]
                trade_pnl_pct = (
                    np.prod(1.0 + trade_returns) - 1.0
                ) * 100.0

                trade_records.append(
                    {
                        "entry_date": dates[entry_idx],
                        "exit_date": dates[i],
                        "side": (
                            "Long" if current_side > 0.0 else "Short"
                        ),
                        "duration_days": i - entry_idx,
                        "pnl_pct": trade_pnl_pct,
                        "is_win": trade_pnl_pct > 0.0,
                        "is_open_at_end": False,
                    }
                )
                in_trade = False

            if not in_trade and current_position != 0.0:
                in_trade = True
                entry_idx = i
                current_side = current_position

        if in_trade:
            last_idx = len(position_values) - 1
            trade_returns = return_values[entry_idx + 1 :]
            trade_pnl_pct = (
                np.prod(1.0 + trade_returns) - 1.0
            ) * 100.0

            trade_records.append(
                {
                    "entry_date": dates[entry_idx],
                    "exit_date": dates[last_idx],
                    "side": (
                        "Long" if current_side > 0.0 else "Short"
                    ),
                    "duration_days": last_idx - entry_idx,
                    "pnl_pct": trade_pnl_pct,
                    "is_win": trade_pnl_pct > 0.0,
                    "is_open_at_end": True,
                }
            )

        trades_df = pd.DataFrame(
            trade_records,
            columns=[
                "entry_date",
                "exit_date",
                "side",
                "duration_days",
                "pnl_pct",
                "is_win",
                "is_open_at_end",
            ],
        )

        trade_logs[label] = trades_df
        total_trades = len(trades_df)

        if total_trades > 0:
            long_trades = int((trades_df["side"] == "Long").sum())
            short_trades = int((trades_df["side"] == "Short").sum())

            winning_trades = trades_df[
                trades_df["pnl_pct"] > 0.0
            ]
            losing_trades = trades_df[
                trades_df["pnl_pct"] < 0.0
            ]

            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            breakeven_count = int(
                (trades_df["pnl_pct"] == 0.0).sum()
            )

            win_rate_pct = win_count / total_trades * 100.0

            average_win_pct = (
                float(winning_trades["pnl_pct"].mean())
                if win_count > 0
                else np.nan
            )
            average_loss_pct = (
                float(losing_trades["pnl_pct"].mean())
                if loss_count > 0
                else np.nan
            )

            payoff_ratio = (
                average_win_pct / abs(average_loss_pct)
                if win_count > 0 and loss_count > 0
                else np.nan
            )
            win_loss_count_ratio = (
                win_count / loss_count
                if loss_count > 0
                else np.nan
            )

            average_duration = float(
                trades_df["duration_days"].mean()
            )
            minimum_duration = int(
                trades_df["duration_days"].min()
            )
            maximum_duration = int(
                trades_df["duration_days"].max()
            )
        else:
            long_trades = 0
            short_trades = 0
            win_count = 0
            loss_count = 0
            breakeven_count = 0
            win_rate_pct = np.nan
            average_win_pct = np.nan
            average_loss_pct = np.nan
            payoff_ratio = np.nan
            win_loss_count_ratio = np.nan
            average_duration = np.nan
            minimum_duration = np.nan
            maximum_duration = np.nan

        trades_per_month = (
            total_trades / months
            if months > 0.0
            else np.nan
        )

        summary_records[label] = {
            "Start date": dates[0].strftime("%Y-%m-%d"),
            "End date": dates[-1].strftime("%Y-%m-%d"),
            "Trading days": len(df),
            "Period (months)": months,
            "Period (years)": years,
            "Gross return (%)": gross_return_pct,
            "Cost drag (%)": cost_drag_pp,
            "Net return (%)": net_return_pct,
            "Net CAGR (%)": cagr_pct,
            "Maximum drawdown (%)": max_drawdown_pct,
            "Annual volatility (%)": annual_volatility_pct,
            "Annual Sharpe ratio": sharpe_ratio,
            "Total trades": total_trades,
            "Long trades": long_trades,
            "Short trades": short_trades,
            "Winning trades": win_count,
            "Losing trades": loss_count,
            "Breakeven trades": breakeven_count,
            "Win rate (%)": win_rate_pct,
            "Average winning trade (%)": average_win_pct,
            "Average losing trade (%)": average_loss_pct,
            "Payoff ratio": payoff_ratio,
            "Win/loss count ratio": win_loss_count_ratio,
            "Average trade duration (days)": average_duration,
            "Minimum trade duration (days)": minimum_duration,
            "Maximum trade duration (days)": maximum_duration,
            "Trades per month": trades_per_month,
        }

    summary_df = pd.DataFrame(summary_records).reindex(metric_names)

    # Format values explicitly so the console table does not depend
    # on pandas display settings.
    integer_metrics = {
        "Trading days",
        "Total trades",
        "Long trades",
        "Short trades",
        "Winning trades",
        "Losing trades",
        "Breakeven trades",
        "Minimum trade duration (days)",
        "Maximum trade duration (days)",
    }

    displayed_values = {}

    for label in labels:
        displayed_values[label] = {}

        for metric in metric_names:
            value = summary_df.at[metric, label]

            if pd.isna(value):
                text = "N/A"
            elif metric in {"Start date", "End date"}:
                text = str(value)
            elif metric in integer_metrics:
                text = str(int(value))
            elif metric == "Average trade duration (days)":
                text = f"{value:.1f}"
            else:
                text = f"{value:.2f}"

            displayed_values[label][metric] = text

    metric_width = max(
        len("Metric"),
        *(len(metric) for metric in metric_names),
    )

    column_widths = {}

    for label in labels:
        column_widths[label] = max(
            len(label),
            *(
                len(displayed_values[label][metric])
                for metric in metric_names
            ),
        )

    header_cells = [
        f"{'Metric':<{metric_width}}",
        *(
            f"{label:>{column_widths[label]}}"
            for label in labels
        ),
    ]

    header = " | ".join(header_cells)
    separator = "-" * len(header)
    mode = "COMPOUNDED" if compounding else "FIXED-SIZE / ADDITIVE"

    print()
    print(f"STRATEGY COMPARISON — {mode}")
    print(separator)
    print(header)
    print(separator)

    section_starts = {
        "Gross return (%)",
        "Total trades",
        "Win rate (%)",
        "Average trade duration (days)",
    }

    for metric in metric_names:
        if metric in section_starts:
            print(separator)

        row_cells = [
            f"{metric:<{metric_width}}",
            *(
                f"{displayed_values[label][metric]:>{column_widths[label]}}"
                for label in labels
            ),
        ]

        print(" | ".join(row_cells))

    print(separator)
    print(
        "N/A = undefined or insufficient data. "
        "Cost drag is gross return minus net return."
    )

    return summary_df, trade_logs
