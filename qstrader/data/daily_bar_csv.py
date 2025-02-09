import functools
import os

import numpy as np
import pandas as pd
import pytz
from qstrader import settings


class CSVDailyBarDataSource(object):
    """
    Encapsulates loading, preparation and querying of CSV files of
    daily 'bar' OHLCV data. The CSV files are converted into a intraday
    timestamped Pandas DataFrame with opening and closing prices.

    Optionally utilises adjusted closing prices (if available) to
    adjust both the close and open.

    Parameters
    ----------
    csv_dir : `str`
        The full path to the directory where the CSV is located.
    asset_type : `str`
        The asset type that the price/volume data is for.
        TODO: Unused at this stage and currently hardcoded to Equity.
    adjust_prices : `Boolean`, optional
        Whether to utilise corporate-action adjusted prices for both
        the open and closing prices. Defaults to True.
    csv_symbols : `list`, optional
        An optional list of CSV symbols to restrict the data source to.
        The alternative is to convert all CSVs found within the
        provided directory.
    """

    def __init__(self, csv_dir, asset_type, adjust_prices=True, csv_symbols=None):
        self.csv_dir = csv_dir
        self.asset_type = asset_type
        self.adjust_prices = adjust_prices
        self.csv_symbols = csv_symbols

        self.asset_bar_frames = self._load_csvs_into_dfs()
        self.asset_bid_ask_frames = self._convert_bars_into_bid_ask_dfs()

    @functools.lru_cache(maxsize=1024 * 1024)
    def get_high(self, dt, asset):
        bar_df = self.asset_bar_frames[asset]
        high_series = bar_df.iloc[bar_df.index.get_indexer([dt], method="pad")]["High"]
        return high_series.iloc[0] if not high_series.empty else np.nan

    @functools.lru_cache(maxsize=1024 * 1024)
    def get_low(self, dt, asset):
        bar_df = self.asset_bar_frames[asset]
        low_series = bar_df.iloc[bar_df.index.get_indexer([dt], method="pad")]["Low"]
        return low_series.iloc[0] if not low_series.empty else np.nan

    @functools.lru_cache(maxsize=1024 * 1024)
    def get_open(self, dt, asset):
        bar_df = self.asset_bar_frames[asset]
        low_series = bar_df.iloc[bar_df.index.get_indexer([dt], method="pad")]["Open"]
        return low_series.iloc[0] if not low_series.empty else np.nan

    def _obtain_asset_csv_files(self):
        """
        Obtain the list of all CSV filenames in the CSV directory.

        Returns
        -------
        `list[str]`
            The list of all CSV filenames.
        """
        return [file for file in os.listdir(self.csv_dir) if file.endswith(".csv")]

    def _obtain_asset_symbol_from_filename(self, csv_file):
        """
        Return the QSTrader symbology for the asset.

        TODO: Remove hardcoding to Equity asset types.

        Parameters
        ----------
        csv_file : `str`
            The name of the CSV file.

        Returns
        -------
        `str`
            The QSTrader symbology of the asset. e.g. 'EQ:SPY'.
        """
        return "EQ:%s" % csv_file.replace(".csv", "")

    def _load_csv_into_df(self, csv_file):
        csv_df = pd.read_csv(
            os.path.join(self.csv_dir, csv_file), index_col="Date", parse_dates=True
        ).sort_index()

        # Ensure all timestamps are set to UTC for consistency
        csv_df = csv_df.set_index(csv_df.index.tz_convert(pytz.UTC))

        # Keep only necessary columns
        csv_df = csv_df[["Open", "High", "Low", "Close", "Adj Close"]]

        return csv_df

    def _load_csvs_into_dfs(self):
        """
        Load all CSVs in the CSV directory into Pandas DataFrames.

        Returns
        -------
        `dict{pd.DataFrame}`
            The asset-symbol keyed dictionary of Pandas DataFrames
            containing the timestamped price/volume data.
        """
        if settings.PRINT_EVENTS:
            print("Loading CSV files into DataFrames...")
        if self.csv_symbols is not None:
            # TODO/NOTE: This assumes existence of CSV symbols
            # within the provided directory.
            csv_files = ["%s.csv" % symbol for symbol in self.csv_symbols]
        else:
            csv_files = self._obtain_asset_csv_files()

        asset_frames = {}
        for csv_file in csv_files:
            asset_symbol = self._obtain_asset_symbol_from_filename(csv_file)
            if settings.PRINT_EVENTS:
                print("Loading CSV file for symbol '%s'..." % asset_symbol)
            csv_df = self._load_csv_into_df(csv_file)
            asset_frames[asset_symbol] = csv_df
        return asset_frames

    def _convert_bar_frame_into_bid_ask_df(self, bar_df):
        bar_df = bar_df.sort_index()

        if self.adjust_prices:
            if "Adj Close" not in bar_df.columns:
                raise ValueError(
                    "Unable to locate Adjusted Close pricing column in CSV data file. "
                    "Prices cannot be adjusted. Exiting."
                )

            bar_df["Adj Open"] = (bar_df["Adj Close"] / bar_df["Close"]) * bar_df[
                "Open"
            ]
            bar_df["Adj High"] = (bar_df["Adj Close"] / bar_df["Close"]) * bar_df[
                "High"
            ]
            bar_df["Adj Low"] = (bar_df["Adj Close"] / bar_df["Close"]) * bar_df["Low"]

            bar_df = bar_df[["Adj Open", "Adj High", "Adj Low", "Adj Close"]]
            bar_df.columns = ["Open", "High", "Low", "Close"]
        else:
            bar_df = bar_df[["Open", "High", "Low", "Close"]]

        seq_df = bar_df.T.unstack(level=0).reset_index()
        seq_df.columns = ["Date", "Market", "Price"]

        # Timestamping Open, High, Low, Close
        seq_df.loc[seq_df["Market"] == "Open", "Date"] += pd.Timedelta(
            hours=14, minutes=30
        )
        seq_df.loc[seq_df["Market"] == "High", "Date"] += pd.Timedelta(
            hours=17, minutes=0
        )
        seq_df.loc[seq_df["Market"] == "Low", "Date"] += pd.Timedelta(
            hours=19, minutes=0
        )
        seq_df.loc[seq_df["Market"] == "Close", "Date"] += pd.Timedelta(
            hours=21, minutes=0
        )

        dp_df = seq_df[["Date", "Price"]]
        dp_df["Bid"] = dp_df["Price"]
        dp_df["Ask"] = dp_df["Price"]
        dp_df = (
            dp_df.loc[:, ["Date", "Bid", "Ask"]].ffill().set_index("Date").sort_index()
        )

        return dp_df

    def _convert_bars_into_bid_ask_dfs(self):
        """
        Convert all of the daily OHLCV 'bar' based DataFrames into
        individually-timestamped open/closing price DataFrames.

        Returns
        -------
        `dict{pd.DataFrame}`
            The converted DataFrames.
        """
        if settings.PRINT_EVENTS:
            print("Adjusting pricing in CSV files...")
        asset_bid_ask_frames = {}
        for asset_symbol, bar_df in self.asset_bar_frames.items():
            if settings.PRINT_EVENTS:
                print("Adjusting CSV file for symbol '%s'..." % asset_symbol)
            asset_bid_ask_frames[asset_symbol] = (
                self._convert_bar_frame_into_bid_ask_df(bar_df)
            )
        return asset_bid_ask_frames

    @functools.lru_cache(maxsize=1024 * 1024)
    def get_bid(self, dt, asset):
        """
        Obtain the bid price of an asset at the provided timestamp.

        Parameters
        ----------
        dt : `pd.Timestamp`
            When to obtain the bid price for.
        asset : `str`
            The asset symbol to obtain the bid price for.

        Returns
        -------
        `float`
            The bid price.
        """
        bid_ask_df = self.asset_bid_ask_frames[asset]
        bid_series = bid_ask_df.iloc[bid_ask_df.index.get_indexer([dt], method="pad")][
            "Bid"
        ]
        try:
            bid = bid_series.iloc[0]
        except KeyError:  # Before start date
            return np.nan
        return bid

    @functools.lru_cache(maxsize=1024 * 1024)
    def get_ask(self, dt, asset):
        """
        Obtain the ask price of an asset at the provided timestamp.

        Parameters
        ----------
        dt : `pd.Timestamp`
            When to obtain the ask price for.
        asset : `str`
            The asset symbol to obtain the ask price for.

        Returns
        -------
        `float`
            The ask price.
        """
        bid_ask_df = self.asset_bid_ask_frames[asset]
        ask_series = bid_ask_df.iloc[bid_ask_df.index.get_indexer([dt], method="pad")][
            "Ask"
        ]
        try:
            ask = ask_series.iloc[0]
        except KeyError:  # Before start date
            return np.nan
        return ask

    def get_assets_historical_closes(self, start_dt, end_dt, assets):
        """
        Obtain a multi-asset historical range of closing prices as a DataFrame,
        indexed by timestamp with asset symbols as columns.

        Parameters
        ----------
        start_dt : `pd.Timestamp`
            The starting datetime of the range to obtain.
        end_dt : `pd.Timestamp`
            The ending datetime of the range to obtain.
        assets : `list[str]`
            The list of asset symbols to obtain closing prices for.

        Returns
        -------
        `pd.DataFrame`
            The multi-asset closing prices DataFrame.
        """
        close_series = []
        for asset in assets:
            if asset in self.asset_bar_frames.keys():
                asset_close_prices = self.asset_bar_frames[asset][["Close"]]
                asset_close_prices.columns = [asset]
                close_series.append(asset_close_prices)

        prices_df = pd.concat(close_series, axis=1).dropna(how="all")
        prices_df = prices_df.loc[start_dt:end_dt]
        return prices_df
