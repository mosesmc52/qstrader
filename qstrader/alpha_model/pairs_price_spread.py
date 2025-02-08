import numpy as np
from qstrader.alpha_model.alpha_model import AlphaModel


class PairsPriceSpreadAlphaModel(AlphaModel):
    def __init__(self, signals, lookback, universe, data_handler):
        self.signals = signals
        self.lookback = lookback
        self.universe = universe
        self.data_handler = data_handler

    def _generate_signals(self, assets, weights, dt):
        z_score, hedge_ratio = self.signals["zscore"](assets, self.lookback)
        z_score = np.clip(z_score, -1, 1)  # Ensure weight is between 0 and 1

        latest_price_0 = self.data_handler.get_asset_latest_mid_price(dt, assets[0])
        latest_price_1 = self.data_handler.get_asset_latest_mid_price(dt, assets[1])

        total_exposure = np.abs(hedge_ratio * latest_price_0 + latest_price_1)

        weights = []
        weights.append((-z_score * -hedge_ratio * latest_price_0) / total_exposure)
        weights.append((-z_score * latest_price_1) / total_exposure)
        weights = {asset: weights[i] for i, asset in enumerate(assets)}

        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= self.lookback:
            weights = self._generate_signals(assets, weights, dt)

        return weights
