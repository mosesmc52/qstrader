import numpy as np
import pandas as pd
from qstrader.signals.signal import Signal


class StatArbPositionCollection(Signal):

    def __init__(
        self,
        start_dt,
        universe,
        lookback,
    ):
        self.lookback = lookback
        super().__init__(start_dt, universe, [lookback])

    def _asset_lookback_key(self, asset):
        """
        Create the buffer dictionary lookup key based
        on asset name and lookback period.

        Parameters
        ----------
        asset : `str`
            The asset symbol name.
        lookback : `int`
            The lookback period.

        Returns
        -------
        `str`
            The lookup key.
        """
        return "%s_%s" % (asset, self.lookback)

    def _calc_positions(self, assets, weights):

        Y_plus = []

        for asset in assets:
            Y_plus.append(
                pd.Series(np.log(self.buffers.prices[self._asset_lookback_key(asset)]))
            )

        log_mkt_val = np.sum(weights * np.column_stack(Y_plus), axis=1)
        num_units = -(log_mkt_val[-1:] - log_mkt_val.mean()) / log_mkt_val.std()
        positions = weights * num_units[0]

        return positions

    def __call__(self, assets, weights):
        """
        For a collection of assets, calculate the johansen weight and use the weight to calculate the
        log zscore.
        """
        if not len(assets):
            return []
        return self._calc_positions(assets, weights)
