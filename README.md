# Pairs Trading & Statistical Arbitrage Pipeline

This repository contains the Final Project for the EPAT (Executive Programme in Algorithmic Trading) course. 
It implements a robust, modular pipeline for backtesting, optimizing, and evaluating pairs trading strategies.

## Project Structure

- `config/`: Contains YAML configuration files to parameterize the simulations.
- `src/engine/`: Core execution engines (Backtester, Analyzer, Optimizer, Meta-Model Selector).
- `src/strategies/`: Concrete pairs trading strategy implementations (Bollinger Bands, Z-Score, Kalman Filter).
- `src/utils/`: Evaluation metrics and plotting functions.
- `notebooks/`: Jupyter notebooks demonstrating single-pair and multi-pair simulations.

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
