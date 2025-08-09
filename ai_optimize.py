"""CLI tool: run backtest, build prompt, send to OpenAI, print suggestions.

Usage:
  OPENAI_API_KEY=sk-... python ai_optimize.py data.csv [--model gpt-4o-mini]
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
from backtesting import Backtest
from src.strategy.divergent_bar import DivergentBar
from src.strategy.strategy_ai import (
    extract_context,
    build_llm_prompt,
    send_prompt_openai,
    parse_suggestions,
    build_condition_diagnostics,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", help="CSV file with Date index & OHLC columns")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--cash", type=int, default=10_000)
    p.add_argument("--commission", type=float, default=0.002)
    p.add_argument("--no-llm", action="store_true", help="Only print prompt & diagnostics; skip LLM call")
    p.add_argument("--suggestions-json", help="Write parsed suggestions JSON to file (if LLM call succeeds)")
    args = p.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Data file not found: {args.data}")

    df = pd.read_csv(args.data, parse_dates=True, index_col='Date')
    bt = Backtest(df, DivergentBar, cash=args.cash, commission=args.commission, finalize_trades=True)
    stats = bt.run()
    # Mirror backtest.py output
    print("\n=== Backtest Stats ===")
    print(stats)
    trade_cols = [
        'Size', 'EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'SL', 'TP',
        'PnL', 'Commission', 'ReturnPct', 'Entry_DivergentBar', 'Exit_DivergentBar'
    ]
    try:
        trades_view = stats['_trades'][trade_cols]
    except Exception:
        trades_view = stats.get('_trades')
    if trades_view is not None:
        print("\n=== Trades ===")
        print(trades_view.to_string())

    # Extract base context (winners/losers/performance) and condition diagnostics
    ctx = extract_context(stats)
    diagnostics = build_condition_diagnostics(bt)
    prompt = build_llm_prompt(ctx)
    if diagnostics:
        import json
        prompt += "\n\nCONDITION_DIAGNOSTICS_JSON:\n" + json.dumps(diagnostics, indent=2)

    print("\n==== Prompt Sent ====")
    print(prompt)
    print("====================\n")

    if args.no_llm:
        print("--no-llm specified: skipping model call.")
        return

    response = send_prompt_openai(prompt, model=args.model)
    if response is None:
        print("No response (missing key or error).")
        return
    print("Raw LLM Response:\n", response)
    suggestions = parse_suggestions(response)
    print("\nParsed Suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s.get('change','')}\n   Rationale: {s.get('rationale','')}\n   Hint: {s.get('implementation_hint','')}")
    if args.suggestions_json:
        try:
            import json, pathlib
            pathlib.Path(args.suggestions_json).write_text(json.dumps(suggestions, indent=2))
            print(f"Suggestions written to {args.suggestions_json}")
        except Exception as e:
            print(f"Failed to write suggestions JSON: {e}")


if __name__ == "__main__":
    main()
