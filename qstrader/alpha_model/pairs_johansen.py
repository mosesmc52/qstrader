import numpy as np
from qstrader.alpha_model.alpha_model import AlphaModel


class PairsJohansenAlphaModel(AlphaModel):
    def __init__(self, signals, lookback, train_len, universe, data_handler):
        self.signals = signals
        self.lookback = lookback
        self.train_len = train_len
        self.universe = universe
        self.data_handler = data_handler

    def _generate_signals(self, dt, assets, weights):
        z_score, hedge_ratio = self.signals["z_score"](assets)

        A = self.data_handler.get_asset_latest_mid_price(dt, assets[0])
        B = self.data_handler.get_asset_latest_mid_price(dt, assets[1])

        total_value = np.sum(np.abs(z_score * hedge_ratio * [A, B]))
        allocations = np.abs(z_score * hedge_ratio * [A, B]) / total_value

        return {asset: allocations[i] for i, asset in enumerate(assets)}

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= max(self.train_len, self.lookback):
            weights = self._generate_signals(dt, assets, weights)

        return weights
