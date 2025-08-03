
# Example OHLC daily data for Google Inc.
import pandas as pd
from backtesting import Backtest
import sys
import os
from src.strategy.sma import SmaCross, SMA
from src.strategy.divergent_bar import DivergentBar



def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]
    input_data = pd.read_csv(filename, parse_dates=True, index_col='Date')
    bt = Backtest(input_data, DivergentBar, cash=10_000, commission=.002, finalize_trades=True)
    stats = bt.run()
    print(stats)
    print(stats['_trades'])
    bt.plot()

if __name__ == "__main__":
    main()