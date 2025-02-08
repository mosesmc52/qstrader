import os

import pandas as pd
import pytz
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.alpha_model.top_nm_momentum import TopNMomentumAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.momentum import MomentumSignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

if __name__ == "__main__":
    # Duration of the backtest
    start_dt = pd.Timestamp("1998-12-22 14:30:00", tz=pytz.UTC)
    burn_in_dt = pd.Timestamp("1999-12-22 14:30:00", tz=pytz.UTC)
    end_dt = pd.Timestamp("2020-12-31 23:59:00", tz=pytz.UTC)

    # Model parameters
    mom_lookback = 126  # Six months worth of business days
    mom_top_n = 3  # Number of assets to include at any one time

    # Construct the symbols and assets necessary for the backtest
    # This utilises the SPDR US sector ETFs, all beginning with XL
    strategy_symbols = ["XL%s" % sector for sector in "BCEFIKPUVY"]
    assets = ["EQ:%s" % symbol for symbol in strategy_symbols]

    # As this is a dynamic universe of assets (XLC is added later)
    # we need to tell QSTrader when XLC can be included. This is
    # achieved using an asset dates dictionary
    asset_dates = {asset: start_dt for asset in assets}

    # udate XLC asset start date when existed on market
    asset_dates["EQ:XLC"] = pd.Timestamp("2018-06-18 00:00:00", tz=pytz.UTC)

    # a dynamic universe is used when stocks are added to the univrse. stocks cannot be removeds
    strategy_universe = DynamicUniverse(asset_dates)

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", "./data")
    strategy_data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=strategy_symbols
    )
    strategy_data_handler = BacktestDataHandler(
        strategy_universe, data_sources=[strategy_data_source]
    )

    # Generate the signals (in this case holding-period return based
    # momentum) used in the top-N momentum alpha model
    momentum = MomentumSignal(start_dt, strategy_universe, lookbacks=[mom_lookback])
    signals = SignalsCollection({"momentum": momentum}, strategy_data_handler)

    # Generate the alpha model instance for the top-N momentum alpha model
    strategy_alpha_model = TopNMomentumAlphaModel(
        signals, mom_lookback, mom_top_n, strategy_universe, strategy_data_handler
    )

    # Construct the strategy backtest and run it
    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
        signals=signals,
        rebalance="end_of_month",
        long_only=True,
        cash_buffer_percentage=0.01,
        burn_in_dt=burn_in_dt,
        data_handler=strategy_data_handler,
    )
    strategy_backtest.run()

    # Construct benchmark assets (buy & hold SPY)
    benchmark_symbols = ["SPY"]
    benchmark_assets = ["EQ:SPY"]
    benchmark_universe = StaticUniverse(benchmark_assets)
    benchmark_data_source = CSVDailyBarDataSource(
        csv_dir, Equity, csv_symbols=benchmark_symbols
    )
    benchmark_data_handler = BacktestDataHandler(
        benchmark_universe, data_sources=[benchmark_data_source]
    )

    # Construct a benchmark Alpha Model that provides
    # 100% static allocation to the SPY ETF, with no rebalance
    benchmark_alpha_model = FixedSignalsAlphaModel({"EQ:SPY": 1.0})
    benchmark_backtest = BacktestTradingSession(
        burn_in_dt,
        end_dt,
        benchmark_universe,
        benchmark_alpha_model,
        rebalance="buy_and_hold",
        long_only=True,
        cash_buffer_percentage=0.01,
        data_handler=benchmark_data_handler,
    )
    benchmark_backtest.run()

    # Performance Output
    tearsheet = TearsheetStatistics(
        strategy_equity=strategy_backtest.get_equity_curve(),
        benchmark_equity=benchmark_backtest.get_equity_curve(),
        title="US Sector Momentum - Top 3 Sectors",
    )
    tearsheet.plot_results()
