import numpy as np
import pandas as pd
from qstrader.signals.signal import Signal
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class JohansenETFLogPriceWeightsCollectionSignal(Signal):

    def __init__(
        self,
        start_dt,
        etf_index,
        universe,
        lookback,
    ):
        self.lookback = lookback
        self.etf_index = etf_index
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

    def _calculate_weights(self, assets):

        etf_price_series = pd.Series(
            self.buffers.prices[self._asset_lookback_key(f"EQ:{self.etf_index}")]
        )

        co_int_assets = []
        co_int_asset_prices = []
        for asset in assets:
            if f"EQ:{self.etf_index}" == asset:
                continue

            asset_price = pd.Series(
                self.buffers.prices[self._asset_lookback_key(asset)]
            )
            Y_test = np.column_stack(
                [
                    np.log(etf_price_series),
                    np.log(asset_price),
                ]
            )
            if np.std(Y_test[:, 1]) > 1e-10:
                result = coint_johansen(Y_test, det_order=0, k_ar_diff=1)
                if result.lr1[0] > result.cvt[0, 0]:
                    co_int_assets.append(asset)
                    co_int_asset_prices.append(asset_price)

        if not len(co_int_asset_prices):
            return [], []

        Y_n = np.column_stack(co_int_asset_prices)

        co_int_asset_prices.append(etf_price_series)
        co_int_assets.append(f"EQ:{self.etf_index}")
        Y_plus = np.column_stack(co_int_asset_prices)
        result = coint_johansen(Y_plus, det_order=0, k_ar_diff=1)

        weights = result.evec[:, 0]

        return co_int_assets, weights

    def __call__(self, assets):
        """ """

        return self._calculate_weights(assets)
