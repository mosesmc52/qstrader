import operator

from qstrader.alpha_model.alpha_model import AlphaModel


class TopNMomentumAlphaModel(AlphaModel):

    def __init__(self, signals, mom_lookback, mom_top_n, universe, data_handler):
        """
        Initialise the TopNMomentumAlphaModel

        Parameters
        ----------
        signals : `SignalsCollection`
            The entity for interfacing with various pre-calculated
            signals. In this instance we want to use 'momentum'.
        mom_lookback : `integer`
            The number of business days to calculate momentum
            lookback over.
        mom_top_n : `integer`
            The number of assets to include in the portfolio,
            ranking from highest momentum descending.
        universe : `Universe`
            The collection of assets utilised for signal generation.
        data_handler : `DataHandler`
            The interface to the CSV data.

        Returns
        -------
        None
        """
        self.signals = signals
        self.mom_lookback = mom_lookback
        self.mom_top_n = mom_top_n
        self.universe = universe
        self.data_handler = data_handler

    def _highest_momentum_asset(self, dt):
        """
        Calculates the ordered list of highest performing momentum
        assets restricted to the 'Top N', for a particular datetime.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the highest momentum assets
            should be calculated.

        Returns
        -------
        `list[str]`
            Ordered list of highest performing momentum assets
            restricted to the 'Top N'.
        """
        assets = self.signals["momentum"].assets

        # Calculate the holding-period return momenta for each asset,
        # for the particular provided momentum lookback period
        all_momenta = {
            asset: self.signals["momentum"](asset, self.mom_lookback)
            for asset in assets
        }

        # Obtain a list of the top performing assets by momentum
        # restricted by the provided number of desired assets to
        # trade per month
        return [
            asset[0]
            for asset in sorted(
                all_momenta.items(), key=operator.itemgetter(1), reverse=True
            )
        ][: self.mom_top_n]

    def _generate_signals(self, dt, weights):
        """
        Calculate the highest performing momentum for each
        asset then assign 1 / N of the signal weight to each
        of these assets.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the signal weights
            should be calculated.
        weights : `dict{str: float}`
            The current signal weights dictionary.

        Returns
        -------
        `dict{str: float}`
            The newly created signal weights dictionary.
        """
        top_assets = self._highest_momentum_asset(dt)

        # there is an equal weight for each asset
        for asset in top_assets:
            weights[asset] = 1.0 / self.mom_top_n
        return weights

    def __call__(self, dt):
        """
        Calculates the signal weights for the top N
        momentum alpha model, assuming that there is
        sufficient data to begin calculating momentum
        on the desired assets.

        Parameters
        ----------
        dt : `pd.Timestamp`
            The datetime for which the signal weights
            should be calculated.

        Returns
        -------
        `dict{str: float}`
            The newly created signal weights dictionary.
        """
        assets = self.universe.get_assets(dt)
        weights = {asset: 0.0 for asset in assets}

        # Only generate weights if the current time exceeds the
        # momentum lookback period
        if self.signals.warmup >= self.mom_lookback:
            weights = self._generate_signals(dt, weights)
        return weights
