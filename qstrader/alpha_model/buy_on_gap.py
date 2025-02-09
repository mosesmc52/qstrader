import numpy as np
import pandas as pd
from qstrader.alpha_model.alpha_model import AlphaModel


class BuyOnGapAlphaModel(AlphaModel):

    def __init__(
        self,
        signals,
        start_dt,
        universe,
        std_lookback,
        ma_lookback,
        top_n,
        entry_z,
        data_handler,
    ):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler
        self.ma_lookback = ma_lookback
        self.std_lookback = std_lookback
        self.top_n = top_n
        self.entry_z = entry_z

    def compute_price(self, low_price, std_return):
        return low_price * (1 - self.entry_z * std_return)

    def compute_ret_gap(self, open_price, yesterday_low_price):
        return (open_price - yesterday_low_price) / yesterday_low_price

    def _generate_signals(self, dt, assets, weights):
        yesterday_dt = dt - pd.tseries.offsets.Day()

        std_returns = self.signals["std_returns"](assets)
        ma = self.signals["ma"](assets)

        buy_price = {}
        return_gap = {}
        accepted_gap = {}
        for asset in assets:

            yesterday_low_price = self.data_handler.get_asset_latest_low_price(
                yesterday_dt, asset
            )
            open_price = self.data_handler.get_asset_latest_open_price(dt, asset)

            buy_price = self.compute_price(yesterday_low_price, std_returns[asset])

            return_gap = self.compute_ret_gap(open_price, yesterday_low_price)

            if open_price < buy_price and open_price > ma[asset]:
                accepted_gap[asset] = return_gap

        if len(accepted_gap):
            accepted_gap_sorted = dict(sorted(accepted_gap.items())[:10])

            weight = 1.0 / len(accepted_gap_sorted)

            for asset, _ in accepted_gap_sorted.items():
                weights[asset] = weight

        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= self.std_lookback:
            weights = self._generate_signals(dt, assets, weights)
        return weights
