import numpy as np
import pandas as pd
import statsmodels.api as sm
from qstrader.signals.signal import Signal


class STDReturnsCollectionSignal(Signal):

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

    def calculate_returns(self, arr):
        return pd.DataFrame(arr).pct_change().values

    def compute_std_returns(self, returns):
        return np.nanstd(returns, ddof=1)  # Use ddof=1 for sample standard deviation

    def _calc_std(self, assets):

        std_ret = {}
        for asset in assets:
            series = pd.Series(self.buffers.prices[self._asset_lookback_key(asset)])[1:]
            returns = self.calculate_returns(series)
            std_ret[asset] = self.compute_std_returns(returns)

        return std_ret

    def __call__(self, assets):
        """
        Calculate the z-score for the asset.
        """

        return self._calc_std(assets)
