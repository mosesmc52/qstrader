# coint_bollinger_strategy.py
import os

import numpy as np
import pandas as pd
import pytz
from qstrader.alpha_model.pairs_price_spread import PairsPriceSpreadAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.price_zscore import PriceZScoreSignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

if __name__ == "__main__":
    start_dt = pd.Timestamp("2006-05-24 00:00:00", tz=pytz.UTC)
    end_dt = pd.Timestamp("2012-04-09 00:00:00", tz=pytz.UTC)

    # Construct the symbols and assets necessary for the backtest
    strategy_symbols = ["GLD", "USO"]
    strategy_assets = ["EQ:%s" % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    # data source to load only those provided symbols
    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", "./data")
    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

    lookback = 20

    zscore = PriceZScoreSignal(start_dt, strategy_universe, lookback=lookback)

    signals = SignalsCollection({"zscore": zscore}, data_handler)

    alpha_model = PairsPriceSpreadAlphaModel(
        signals, lookback, strategy_universe, data_handler
    )

    # Construct the strategy backtest and run it
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        alpha_model,
        signals=signals,
        rebalance="daily",
        cash_buffer_percentage=0.00,
        gross_leverage=1.0,
        data_handler=data_handler,
    )
    strategy_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        title="Pairs Price Spread",
    )
    tearsheet.plot_results()
