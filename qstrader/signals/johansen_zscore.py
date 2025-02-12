import numpy as np
import pandas as pd
import statsmodels.tsa.vector_ar.vecm as vecm
from qstrader.signals.signal import Signal


class JohasenZScore(Signal):

    def __init__(self, start_dt, universe, lookback, train_len):
        self.train_len = train_len
        super().__init__(start_dt, universe, [train_len, lookback])

    def _calc_hedge_ratio(self, assets):

        A = pd.Series(self.buffers.prices[f"{assets[0]}_{self.train_len}"]).values

        B = pd.Series(self.buffers.prices[f"{assets[1]}_{self.train_len}"]).values

        Y_test = np.column_stack((A, B))

        results = vecm.coint_johansen(Y_test, det_order=0, k_ar_diff=1)
        hedge_ratio = results.evec[:, 0]
        Y_port = np.sum(Y_test[-15:, :] * hedge_ratio, axis=1)
        ma, mstd = np.mean(Y_port), np.std(Y_port)
        return -(Y_port[-1] - ma) / mstd, hedge_ratio

    def __call__(self, assets):
        """
        Calculate the z-score for the asset.
        """

        return self._calc_hedge_ratio(assets)
