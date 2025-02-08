import numpy as np
import pandas as pd
import statsmodels.api as sm
from qstrader.signals.signal import Signal


class LogZScoreSignal(Signal):

    def __init__(
        self,
        start_dt,
        universe,
        lookback,
    ):
        self.lookback = lookback
        super().__init__(start_dt, universe, [lookback])

    @staticmethod
    def _asset_lookback_key(asset, lookback):
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
        return "%s_%s" % (asset, lookback)

    def _z_score(self, assets, lookback):

        X_prices = pd.Series(
            np.log(
                self.buffers.prices[
                    LogZScoreSignal._asset_lookback_key(assets[0], self.lookback)
                ]
            )
        )

        Y_prices = pd.Series(
            np.log(
                self.buffers.prices[
                    LogZScoreSignal._asset_lookback_key(assets[1], self.lookback)
                ]
            )
        )

        X = sm.add_constant(X_prices)
        model = sm.OLS(Y_prices, X).fit()

        hedge_ratio = model.params[0]

        residuals = []
        for i in range(0, len(X_prices)):
            residuals.append(-hedge_ratio * X_prices[i] + Y_prices[i])

        latest_residual = residuals[0]
        return (latest_residual - np.mean(residuals)) / np.std(residuals), hedge_ratio

    def __call__(self, assets, lookback):
        """
        Calculate the z-score for the asset.
        """

        return self._z_score(assets, lookback)
