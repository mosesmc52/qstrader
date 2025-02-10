import numpy as np


class BacktestDataHandler(object):
    """ """

    def __init__(self, universe, data_sources=None):
        self.universe = universe
        self.data_sources = data_sources

    def get_asset_latest_bid_price(self, dt, asset_symbol):
        """ """
        bid = np.nan

        for ds in self.data_sources:
            try:
                bid = ds.get_bid(dt, asset_symbol)
                if not np.isnan(bid):
                    return bid
            except Exception:
                bid = np.nan
        return bid

    def get_asset_latest_ask_price(self, dt, asset_symbol):
        """ """
        ask = np.nan
        for ds in self.data_sources:
            try:
                ask = ds.get_ask(dt, asset_symbol)
                if not np.isnan(ask):
                    return ask
            except Exception:
                ask = np.nan
        return ask

    def get_asset_latest_bid_ask_price(self, dt, asset_symbol):
        """ """
        bid = self.get_asset_latest_bid_price(dt, asset_symbol)
        return (bid, bid)

    def get_asset_latest_mid_price(self, dt, asset_symbol):
        """ """
        bid_ask = self.get_asset_latest_bid_ask_price(dt, asset_symbol)
        try:
            mid = (bid_ask[0] + bid_ask[1]) / 2.0
        except Exception:
            mid = np.nan
        return mid

    def get_asset_latest_high_price(self, dt, asset_symbol):
        """Retrieve the latest high price for an asset."""
        high = np.nan
        for ds in self.data_sources:
            try:
                high = ds.get_high(dt, asset_symbol)
                if not np.isnan(high):
                    return high
            except Exception:
                high = np.nan
        return high

    def get_asset_latest_low_price(self, dt, asset_symbol):
        """Retrieve the latest low price for an asset."""
        low = np.nan
        for ds in self.data_sources:
            try:
                low = ds.get_low(dt, asset_symbol)
                if not np.isnan(low):
                    return low
            except Exception:
                low = np.nan
        return low

    def get_asset_latest_open_price(self, dt, asset_symbol):
        """Retrieve the latest open price for an asset."""
        open = np.nan
        for ds in self.data_sources:
            try:

                open = ds.get_open(dt, asset_symbol)
                if not np.isnan(open):
                    return open
            except Exception:
                open = np.nan
        return open

    def get_assets_historical_range_close_price(self, start_dt, end_dt, asset_symbols):
        """ """
        prices_df = None
        for ds in self.data_sources:
            try:
                prices_df = ds.get_assets_historical_closes(
                    start_dt, end_dt, asset_symbols
                )
                if prices_df is not None:
                    return prices_df
            except Exception:
                raise
        return prices_df
