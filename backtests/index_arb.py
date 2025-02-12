import os

import pandas as pd
import pytz
from qstrader.alpha_model.fixed_signals import FixedSignalsAlphaModel
from qstrader.alpha_model.stat_arb_etf import StatArbETFAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.dynamic import DynamicUniverse
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.arb_position_collection import StatArbPositionCollection
from qstrader.signals.johansen_etf_log_price_weights_collection import (
    JohansenETFLogPriceWeightsCollectionSignal,
)
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

if __name__ == "__main__":
    # parameters
    etf_index = "XOP"
    lookback = 5
    train_len = 250

    start_dt = pd.Timestamp("2010-01-31 14:30:00", tz=pytz.UTC)
    end_dt = pd.Timestamp("2015-01-01 23:59:00", tz=pytz.UTC)

    # Construct the symbols and assets necessary for the backtest
    strategy_symbols = [
        "XOP",
        "AR",
        "EQT",
        "TPL",
        "CTRA",
        "RRC",
        "EXE",
        "VLO",
        "OVV",
        "MPC",
        "HES",
        "EOG",
        "CIVI",
        "MTDR",
        "COP",
        "FANG",
        "CVX",
        "OXY",
        "XOM",
        "APA",
        "PSX",
        "DVN",
        "PR",
        "DINO",
        "CHRD",
        "SM",
        "VNOM",
        "MUR",
        "PBF",
        "CNX",
        "MGY",
        "NOG",
        "GPOR",
        "CRC",
        "CRGY",
        "SOC",
        "KOS",
        "CRK",
        "WKC",
        "TALO",
        "DK",
        "PARR",
        "VTLE",
        "STR",
        "CLMT",
        "CVI",
        "GPRE",
        "BRY",
        "CLNE",
        "REX",
        "EGY",
    ]

    strategy_assets = ["EQ:%s" % symbol for symbol in strategy_symbols]
    strategy_universe = StaticUniverse(strategy_assets)

    # data source to load only those provided symbols
    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", "./data")
    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

    johansen_log_price_weights = JohansenETFLogPriceWeightsCollectionSignal(
        start_dt, etf_index, strategy_universe, lookback=train_len
    )

    stat_arb_position = StatArbPositionCollection(
        start_dt, strategy_universe, lookback=lookback
    )

    signals = SignalsCollection(
        {
            "johansen_log_price_weights": johansen_log_price_weights,
            "stat_arb_position": stat_arb_position,
        },
        data_handler,
    )

    strategy_alpha_model = StatArbETFAlphaModel(
        signals, strategy_universe, etf_index, train_len, lookback, data_handler
    )

    strategy_backtest = BacktestTradingSession(
        start_dt,
        end_dt,
        strategy_universe,
        strategy_alpha_model,
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
        title="Stat Arbritrage ETF",
    )
    tearsheet.plot_results()
