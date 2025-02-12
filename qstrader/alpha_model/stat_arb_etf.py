import numpy as np
from qstrader.alpha_model.alpha_model import AlphaModel


class StatArbETFAlphaModel(AlphaModel):

    def __init__(
        self,
        signals,
        universe,
        etf_index,
        train_len,
        lookback,
        data_handler=None,
    ):
        self.universe = universe
        self.signals = signals
        self.data_handler = data_handler
        self.etf_index = etf_index
        self.lookback = lookback
        self.train_len = train_len

    def _generate_signals(self, assets, weights, dt):
        coint_assets, coint_weights = self.signals["johansen_log_price_weights"](assets)
        if len(coint_assets):
            positions = self.signals["stat_arb_position"](coint_assets, coint_weights)

            portfolio_weights = positions / np.sum(np.abs(positions))

            weights = {
                asset: portfolio_weights[i] for i, asset in enumerate(coint_assets)
            }

        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= self.train_len:
            weights = self._generate_signals(assets, weights, dt)

        return weights
