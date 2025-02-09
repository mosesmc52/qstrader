import os

import pandas as pd
import pytz
from qstrader.alpha_model.buy_on_gap import BuyOnGapAlphaModel
from qstrader.asset.equity import Equity
from qstrader.asset.universe.static import StaticUniverse
from qstrader.data.backtest_data_handler import BacktestDataHandler
from qstrader.data.daily_bar_csv import CSVDailyBarDataSource
from qstrader.signals.ma_collection import MACollectionSignal
from qstrader.signals.signals_collection import SignalsCollection
from qstrader.signals.std_retuns_collection import STDReturnsCollectionSignal
from qstrader.statistics.tearsheet import TearsheetStatistics
from qstrader.trading.backtest import BacktestTradingSession

if __name__ == "__main__":
    start_dt = pd.Timestamp("2008-01-31 14:30:00", tz=pytz.UTC)
    end_dt = pd.Timestamp("2009-01-01 23:59:00", tz=pytz.UTC)

    # Construct the symbols and assets necessary for the backtest
    strategy_symbols = [
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

    # To avoid loading all CSV files in the directory, set the
    # data source to load only those provided symbols
    csv_dir = os.environ.get("QSTRADER_CSV_DATA_DIR", "./data")
    data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols)
    data_handler = BacktestDataHandler(strategy_universe, data_sources=[data_source])

    ma_lookback = 20
    std_lookback = 90
    top_n = 10
    entry_z = 1

    std_returns = STDReturnsCollectionSignal(start_dt, strategy_universe, std_lookback)
    ma = MACollectionSignal(start_dt, strategy_universe, ma_lookback)
    signals = SignalsCollection({"std_returns": std_returns, "ma": ma}, data_handler)

    alpha_model = BuyOnGapAlphaModel(
        signals,
        start_dt,
        strategy_universe,
        std_lookback,
        ma_lookback,
        top_n,
        entry_z,
        data_handler,
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
        title="Buy on Gap",
    )
    tearsheet.plot_results()
