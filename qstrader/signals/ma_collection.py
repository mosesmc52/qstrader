import numpy as np
import pandas as pd
import statsmodels.api as sm
from qstrader.signals.signal import Signal


class MACollectionSignal(Signal):

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

    def _calc_ma(self, assets):

        ma = {}
        for asset in assets:
            series = pd.Series(self.buffers.prices[self._asset_lookback_key(asset)])[1:]
            ma[asset] = series.mean()

        return ma

    def __call__(self, assets):
        """
        Calculate the z-score for the asset.
        """

        return self._calc_ma(assets)
