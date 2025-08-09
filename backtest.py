
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
    columns = [
        'Size', 'EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'SL', 'TP',
        'PnL', 'Commission', 'ReturnPct', 'Entry_DivergentBar', 'Exit_DivergentBar'
    ]
    print(stats['_trades'][columns].to_string())
    with open("results/trades_output.txt", "w") as f:
        f.write(stats['_trades'][columns].to_string())

    bt.plot()

if __name__ == "__main__":
    main()