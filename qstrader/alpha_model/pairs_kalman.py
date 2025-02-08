import numpy as np
from qstrader.alpha_model.alpha_model import AlphaModel


class PairsKalmanTradingStrategy(AlphaModel):
    def __init__(self, signals, universe, data_handler):
        self.signals = signals
        self.universe = universe
        self.data_handler = data_handler
        self.invested = False

    def _generate_signals(self, assets, weights):

        hedge_ratio, Q, error = self.signals["innovation_variance"](assets)

        sqrt_Q = np.sqrt(Q)
        if not self.invested:
            if error < -sqrt_Q:
                self.num_units = 1  # Enter long
                self.invested = True
            elif error > sqrt_Q:
                self.num_units = -1  # Enter short
                self.invested = True
        else:
            if error > -sqrt_Q and self.num_units == 1:
                self.num_units = 0  # Exit long
                self.invested = False
            elif error < sqrt_Q and self.num_units == -1:
                self.num_units = 0  # Exit short
                self.invested = False

        # The first asset (x) gets -hedge_ratio (short position).
        # The second asset (y) gets 1 (long position).
        allocation = np.array([-hedge_ratio, 1])
        #
        # # normalizing weights between 0 and 1
        allocation /= np.sum(np.abs(allocation))
        #
        # # # assigning weights
        weights = {
            asset: allocation[i] * self.num_units for i, asset in enumerate(assets)
        }

        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        weights = self._generate_signals(assets, weights)

        return weights
