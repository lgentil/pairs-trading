import os
import glob
from typing import Optional, Union, Dict
import pandas as pd


class MQLDataLoader:
    """Historical data loader and preprocessor for MetaTrader 5 (MQL5) exports.

    Reads intraday CSV files (e.g. M1 timeframe), normalizes timestamps into a unified DatetimeIndex,
    synchronizes asset pairs via inner join, and applies canonical OHLCV resampling rules
    to achieve the target aggregation timeframe:
      - Open: first
      - High: max
      - Low: min
      - Close: last
      - TickVol: sum
      - Vol: sum
      - Spread: last

    Attributes:
        data_dir (str): Base directory containing MetaTrader CSV exports.
    """

    RESAMPLE_RULES: Dict[str, str] = {
        "<OPEN>": "first",
        "<HIGH>": "max",
        "<LOW>": "min",
        "<CLOSE>": "last",
        "<TICKVOL>": "sum",
        "<VOL>": "sum",
        "<SPREAD>": "last",
    }

    def __init__(self, data_dir: str = "data/data_mql") -> None:
        """Initializes the MQLDataLoader.

        Args:
            data_dir (str): Base directory path storing MetaTrader CSV files.
        """
        self.data_dir: str = data_dir

    def _clean_ticker(self, ticker: str) -> str:
        """Strips exchange suffixes (such as '.SA') to match the MetaTrader naming convention."""
        return ticker.upper().replace(".SA", "").strip()

    def find_file_for_ticker(self, ticker: str) -> str:
        """Locates the CSV file matching the specified ticker symbol in the data directory.

        Args:
            ticker (str): Asset ticker symbol (e.g., 'PETR3' or 'PETR3.SA').

        Returns:
            str: Resolved relative or absolute path to the matching CSV file.

        Raises:
            FileNotFoundError: If no matching file is found in the target directory.
        """
        clean_name = self._clean_ticker(ticker)
        pattern = os.path.join(self.data_dir, f"{clean_name}_*.csv")
        matching_files = glob.glob(pattern)

        if not matching_files:
            # Fallback to case-insensitive scan in directory
            all_files = os.listdir(self.data_dir) if os.path.exists(self.data_dir) else []
            for f in all_files:
                if f.upper().startswith(f"{clean_name}_") and f.lower().endswith(".csv"):
                    return os.path.join(self.data_dir, f)
            raise FileNotFoundError(
                f"No CSV file found for ticker '{ticker}' in directory '{self.data_dir}'. "
                f"Expected naming pattern: {clean_name}_*.csv"
            )

        return matching_files[0]

    def load_single_asset(
        self,
        ticker: str,
        timeframe: Optional[str] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None
    ) -> pd.DataFrame:
        """Loads and resamples full OHLCV market columns for an individual asset.

        Args:
            ticker (str): Asset ticker symbol (e.g., 'PETR3').
            timeframe (Optional[str]): Resample rule string (e.g., '5min', '15min', '1h').
            start_date (Optional[Union[str, pd.Timestamp]]): Optional start boundary filter.
            end_date (Optional[Union[str, pd.Timestamp]]): Optional end boundary filter.

        Returns:
            pd.DataFrame: Formatted DataFrame with ticker-prefixed columns and DatetimeIndex.
        """
        filepath = self.find_file_for_ticker(ticker)

        df = pd.read_csv(
            filepath,
            sep="\t",
            dtype={
                "<DATE>": str,
                "<TIME>": str,
                "<OPEN>": float,
                "<HIGH>": float,
                "<LOW>": float,
                "<CLOSE>": float,
                "<TICKVOL>": float,
                "<VOL>": float,
                "<SPREAD>": float,
            }
        )

        # Vectorized DatetimeIndex construction
        dt_index = pd.to_datetime(
            df["<DATE>"] + " " + df["<TIME>"],
            format="%Y.%m.%d %H:%M:%S"
        )
        df.index = dt_index
        cols_to_drop = [c for c in ["<DATE>", "<TIME>"] if c in df.columns]
        df = df.drop(columns=cols_to_drop).sort_index()

        # Date boundary filtering before resampling for performance optimization
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            if end_dt.time() == pd.Timestamp("00:00:00").time():
                end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            df = df[df.index <= end_dt]

        # Resample with canonical market aggregation rules
        if timeframe is not None and str(timeframe).strip() != "":
            tf_rule = self._normalize_timeframe(timeframe)
            if tf_rule not in ["1min", "1m"]:
                agg_rules = {col: self.RESAMPLE_RULES[col] for col in df.columns if col in self.RESAMPLE_RULES}
                df = df.resample(tf_rule).agg(agg_rules).dropna(subset=["<CLOSE>"])

        # Column mapping:
        # <CLOSE> retains the clean ticker name for backward compatibility with estimation models.
        # Additional OHLCV features receive designated suffixes (e.g., PETR3_open, PETR3_high).
        col_rename = {
            "<CLOSE>": ticker,
            "<OPEN>": f"{ticker}_open",
            "<HIGH>": f"{ticker}_high",
            "<LOW>": f"{ticker}_low",
            "<VOL>": f"{ticker}_vol",
            "<TICKVOL>": f"{ticker}_tickvol",
            "<SPREAD>": f"{ticker}_spread",
        }
        df = df.rename(columns=col_rename)
        return df

    def load_pair(
        self,
        ticker_x: str,
        ticker_y: str,
        timeframe: Optional[str] = None,
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """Loads and aligns an asset pair with canonical OHLCV resampling.

        Args:
            ticker_x (str): Ticker symbol for Asset X (independent variable).
            ticker_y (str): Ticker symbol for Asset Y (dependent variable).
            timeframe (Optional[str]): Resample rule string (e.g., '5min', '15min', '60min').
            start_date (Optional[Union[str, pd.Timestamp]]): Optional start boundary filter.
            end_date (Optional[Union[str, pd.Timestamp]]): Optional end boundary filter.
            verbose (bool): If True, outputs loading statistics to stdout.

        Returns:
            pd.DataFrame: Synchronized DataFrame containing all OHLCV columns for both assets.
        """
        if verbose:
            print(f"[*] Loading complete MQL OHLCV market data for {ticker_x} and {ticker_y}...")

        df_x = self.load_single_asset(ticker_x, timeframe=timeframe, start_date=start_date, end_date=end_date)
        df_y = self.load_single_asset(ticker_y, timeframe=timeframe, start_date=start_date, end_date=end_date)

        # Synchronize timestamps via inner join (keeping bars where both assets traded)
        df_pair = pd.concat([df_x, df_y], axis=1, join="inner").dropna(subset=[ticker_x, ticker_y])

        if verbose:
            tf_desc = self._normalize_timeframe(timeframe) if timeframe else "M1"
            print(f"[*] Resample '{tf_desc}' completed successfully: {len(df_pair)} bars | Period: {df_pair.index[0]} to {df_pair.index[-1]}")

        return df_pair

    @staticmethod
    def _normalize_timeframe(tf: Union[str, int]) -> str:
        """Normalizes user-friendly timeframe abbreviations and MetaTrader period codes
        to valid Pandas resample frequency rules.

        Supports all MetaTrader 5 standard and custom timeframes:
          - Minutes: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30 (M1 to M30)
          - Hours: H1 (60min), H2, H3, H4, H6, H8, H12
          - Days/Weeks/Months: D1, W1, MN1
        """
        if tf is None:
            return "1min"

        tf_str = str(tf).strip().lower()

        # MetaTrader specific aliases
        mapping = {
            # Minutes
            "1": "1min", "1m": "1min", "m1": "1min", "1min": "1min",
            "2": "2min", "2m": "2min", "m2": "2min", "2min": "2min",
            "3": "3min", "3m": "3min", "m3": "3min", "3min": "3min",
            "4": "4min", "4m": "4min", "m4": "4min", "4min": "4min",
            "5": "5min", "5m": "5min", "m5": "5min", "5min": "5min",
            "6": "6min", "6m": "6min", "m6": "6min", "6min": "6min",
            "10": "10min", "10m": "10min", "m10": "10min", "10min": "10min",
            "12": "12min", "12m": "12min", "m12": "12min", "12min": "12min",
            "15": "15min", "15m": "15min", "m15": "15min", "15min": "15min",
            "20": "20min", "20m": "20min", "m20": "20min", "20min": "20min",
            "30": "30min", "30m": "30min", "m30": "30min", "30min": "30min",

            # Hours
            "60": "1h", "60m": "1h", "m60": "1h", "60min": "1h",
            "h1": "1h", "1h": "1h",
            "h2": "2h", "2h": "2h", "120m": "2h", "120min": "2h",
            "h3": "3h", "3h": "3h", "180m": "3h", "180min": "3h",
            "h4": "4h", "4h": "4h", "240m": "4h", "240min": "4h",
            "h6": "6h", "6h": "6h", "360m": "6h", "360min": "6h",
            "h8": "8h", "8h": "8h", "480m": "8h", "480min": "8h",
            "h12": "12h", "12h": "12h", "720m": "12h", "720min": "12h",

            # Days / Weeks / Months
            "d1": "1D", "1d": "1D", "d": "1D",
            "w1": "1W", "1w": "1W", "w": "1W",
            "mn1": "1MS", "1mn": "1MS", "mn": "1MS",
        }

        if tf_str in mapping:
            return mapping[tf_str]

        # Generic pattern parsing:
        # e.g., 'm2', 'm15' -> '2min', '15min'
        if tf_str.startswith("m") and tf_str[1:].isdigit():
            return f"{tf_str[1:]}min"
        # e.g., 'h1', 'h4' -> '1h', '4h'
        if tf_str.startswith("h") and tf_str[1:].isdigit():
            return f"{tf_str[1:]}h"
        # e.g., '2m', '10m' -> '2min', '10min'
        if tf_str.endswith("m") and tf_str[:-1].isdigit():
            return f"{tf_str[:-1]}min"
        # e.g., '2', '5', '12' -> '2min', '5min', '12min'
        if tf_str.isdigit():
            return f"{tf_str}min"

        return tf_str
