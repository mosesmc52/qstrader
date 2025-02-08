import numpy as np
from qstrader.alpha_model.alpha_model import AlphaModel


class PairsBollingerAlphaModel(AlphaModel):
    def __init__(self, signals, lookback, entry_z, exit_z, universe, data_handler):
        self.signals = signals
        self.lookback = lookback
        self.universe = universe
        self.data_handler = data_handler
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.num_units = 0
        self.invested = False

    def _generate_signals(self, assets, weights, dt):
        z_score, hedge_ratio = self.signals["zscore"](assets, self.lookback)

        if not self.invested:
            if z_score < -self.entry_z:
                self.num_units = 1  # Enter long
                self.invested = True
            elif z_score > self.entry_z:
                self.num_units = -1  # Enter short
                self.invested = True
        else:
            if z_score > -self.exit_z and self.num_units == 1:
                self.num_units = 0  # Exit long
                self.invested = False
            elif z_score < self.exit_z and self.num_units == -1:
                self.num_units = 0  # Exit short
                self.invested = False

        # The first asset (x) gets -hedge_ratio (short position).
        # The second asset (y) gets 1 (long position).
        allocation = np.array([-hedge_ratio, 1])

        # normalizing weights between 0 and 1
        allocation /= np.sum(np.abs(allocation))

        # # assigning weights
        weights = {
            asset: allocation[i] * self.num_units for i, asset in enumerate(assets)
        }

        return weights

    def __call__(self, dt):
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        if self.signals.warmup >= self.lookback:
            weights = self._generate_signals(assets, weights, dt)

        return weights
