import pandas as pd
import argparse
import os

# Entry condition: volume spike + bullish candle
def filtered_entry_signal(df, volume_threshold=1.0):
    avg_volume = df['Volume'].shift(1).rolling(4).mean()
    return (df['Volume'] > avg_volume * volume_threshold) & (df['Close'] > df['Open'])

# Backtest logic with next-day open entry, TP, trailing stop, and 3-day hold
def backtest_with_details(df, signal_mask, hold_days=3, tp_pct=0.03, trailing_pct=0.005):
    df = df.reset_index(drop=True)
    df['Entry'] = None
    df['Exit'] = None
    df['PnL'] = None
    df['Outcome'] = None

    for i in range(len(df) - hold_days - 1):  # ensure room for next-day open and hold_days
        if signal_mask[i]:
            if pd.isna(df.at[i + 1, 'Open']):
                continue
            entry_date = df.at[i + 1, 'Date']
            entry_price = df.at[i + 1, 'Open']
            take_profit = entry_price * (1 + tp_pct)
            trailing_stop = entry_price * (1 - trailing_pct)
            max_price = entry_price
            outcome = 'HOLD'
            exit_price = entry_price

            for j in range(1, hold_days + 1):
                idx = i + 1 + j
                if idx >= len(df):
                    break
                high = df.at[idx, 'High']
                low = df.at[idx, 'Low']
                close = df.at[idx, 'Close']

                max_price = max(max_price, high)
                trailing_stop = max(trailing_stop, max_price * (1 - trailing_pct))

                if high >= take_profit:
                    outcome = 'TP'
                    exit_price = take_profit
                    break
                elif low <= trailing_stop:
                    outcome = 'TRAIL_STOP'
                    exit_price = trailing_stop
                    break
                elif j == hold_days:
                    outcome = 'HOLD'
                    exit_price = close

            pnl = round(exit_price - entry_price, 2)
            df.at[i + 1, 'Entry'] = entry_price
            df.at[i + 1, 'Exit'] = exit_price
            df.at[i + 1, 'PnL'] = pnl
            df.at[i + 1, 'Outcome'] = outcome

    df.dropna(subset=['Entry'], inplace=True)
    return df

def clean_and_prepare(df):
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'Date'}, inplace=True)
    df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]
    df['Date'] = pd.to_datetime(df['Date'])
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def batch_test(input_dir, output_dir):
    """
    Run batch backtesting on all CSV files in input_dir, collect results, and save combined results to output_dir.
    Prints summary for each file.
    """
    os.makedirs(output_dir, exist_ok=True)
    combined_results = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.csv'):
            input_path = os.path.join(input_dir, fname)
            try:
                df = pd.read_csv(input_path)
                df = clean_and_prepare(df)
                signal = filtered_entry_signal(df, volume_threshold=1.1)
                result = backtest_with_details(df, signal)
                symbol = os.path.splitext(fname)[0]
                result['Symbol'] = symbol
                combined_results.append(result)
                print(f"{symbol}: Total Trades={len(result)}, Cumulative PnL={round(result['PnL'].sum(), 2) if not result.empty else 0.0}")
            except Exception as e:
                print(f"Error processing {fname}: {e}")
    if combined_results:
        final_df = pd.concat(combined_results, ignore_index=True)
        final_df.to_csv(os.path.join(output_dir, "combined_results.csv"), index=False)


def main(input_file, output_file):
    df = pd.read_csv(input_file)
    df = clean_and_prepare(df)
    signal = filtered_entry_signal(df, volume_threshold=1.1)
    result = backtest_with_details(df, signal)
    result.to_csv(output_file, index=False)
    summary = {
        "file": os.path.basename(input_file),
        "total_trades": len(result),
        "cumulative_PnL": round(result['PnL'].sum(), 2) if not result.empty else 0.0
    }
    print(f"{summary['file']}: Total Trades={summary['total_trades']}, Cumulative PnL={summary['cumulative_PnL']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Volume-based trading strategy using next-day open price for entry.")
    parser.add_argument("input", nargs="?", help="Path to input CSV file with OHLCV data")
    parser.add_argument("output", nargs="?", help="Path to output CSV file for results")
    parser.add_argument("--batch", action="store_true", help="Run batch mode on input_dir/output_dir")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing input CSV files for batch mode")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output CSV files for batch mode")
    args = parser.parse_args()

    if args.batch:
        if not args.input_dir or not args.output_dir:
            print("Batch mode requires --input_dir and --output_dir.")
        else:
            batch_test(args.input_dir, args.output_dir)
    else:
        if not args.input or not args.output:
            print("Single file mode requires input and output file arguments.")
        else:
            main(args.input, args.output)
