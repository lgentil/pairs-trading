# Pairs Trading & Statistical Arbitrage Pipeline

This repository contains the Final Project for the EPAT (Executive Programme in Algorithmic Trading - Batch 70) course. 
It implements a robust, modular pipeline for backtesting, optimizing, and evaluating pairs trading strategies.

## Project Abstract

This quantitative trading research project presents a comparative study of mean-reversion and statistical arbitrage strategies applied to Brazilian equities listed on the B3 exchange. The primary objective is to evaluate the structural dynamics of asset spreads, address non-stationarity risks, and build an adaptive auto-selection meta-model capable of dynamically routing capital to the most robust quantitative model under prevailing market regimes.

The framework evaluates three distinct strategy architectures across a 15-year historical dataset (70% In-Sample training / 30% Out-of-Sample forward validation):
1. Dynamic Mean Reversion (Bollinger Bands): Combines dynamic lookback moving averages derived from the Ornstein-Uhlenbeck (OU) half-life with rolling Ordinary Least Squares (OLS) hedge ratios.
2. Dynamic Z-Score Strategy: Normalizes rolling spread deviations into dynamic Z-Score signals with parameterized entry, exit, and stop-loss boundaries.
3. State-Space Kalman Filter Strategy: Employs state-space recursive estimation for time-varying hedge ratios and dynamic observation variance scaling.

To overcome performance degradation during non-stationary spread regimes, a Statistical Regime Auto-Selector (governed by rolling Augmented Dickey-Fuller p-values and half-life thresholds) was developed. Backtest executions on the primary benchmark pair Itaú Unibanco (ITUB3.SA) vs. Itaúsa (ITSA3.SA) incorporated a realistic 0.08% round-trip transaction cost drag. The results demonstrated that the meta-models effectively stabilize equity growth, reducing annualized volatility from 5.26% down to 3.73% while increasing the risk-adjusted Sharpe ratio from 1.69 to 2.29. Furthermore, an adaptive volatility-targeted position sizing framework (up to 4.0x dynamic leverage) was established, yielding net cumulative out-of-sample returns of 323.61% at 18.41% annualized volatility for the Bollinger Bands Strategy. Similar results were achieved for other strategies.


## Benchmark

To compare the strategies against the benchmark, the Quantstats library (Python) was used to generate two reports for each strategy: one using the independent asset (ticker_x) as the benchmark, and the other using the CDI (a risk-free investment).The reports are available in the `quantstats_report` folder.

## Note

The data price used to run the simulations for all asset pairs is saved in the `data` folder of this repository. Yahoo Finance may occasionally modify the data due to dividends, reverse splits, or stock splits that could occur in the future.

## Project Structure

- `config/`: Contains YAML configuration files to parameterize the simulations.
- `src/engine/`: Core execution engines (Backtester, Analyzer, Optimizer, Meta-Model Selector).
- `src/strategies/`: Concrete pairs trading strategy implementations (Bollinger Bands, Z-Score, Kalman Filter).
- `src/utils/`: Evaluation metrics and plotting functions.
- `notebooks/`: Jupyter notebooks demonstrating single-pair and multi-pair simulations.
- `data/`: repository of all asset pairs data price in .csv


## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

You can use the notebooks in the `notebooks/` directory to run the simulations.
- `01_single_pair_simulation.ipynb`: Evaluates strategies on a specific pair.
- `02_multi_pair_simulation.ipynb`: Executes backtests on a universe of pre-defined pairs.

Modify `config/settings.yaml` to change the pairs, hyperparameter grids, or backtesting parameters.

## Contact

**Luiz Alberto Gentil Mendes**
- LinkedIn: [Luiz Gentil](https://www.linkedin.com/in/luiz-gentil-404584/)


