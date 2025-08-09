"""LLM-driven strategy improvement scaffolding.

This module provides utilities to:
1. Extract structured trade & performance data from a backtest run.
2. Build a prompt for an LLM to suggest rule modifications.
3. Parse structured JSON suggestions.
4. (Placeholder) Apply modifications (left for manual validation to avoid unsafe dynamic code).

Usage outline:
    from backtesting import Backtest
    from .divergent_bar import DivergentBar
    from .strategy_ai import build_llm_prompt, extract_context

    bt = Backtest(data, DivergentBar, cash=10_000, commission=.002)
    stats = bt.run()
    ctx = extract_context(stats, max_trades=30)
    prompt = build_llm_prompt(ctx)
    # send `prompt` to your LLM client (OpenAI, Anthropic, etc.)

Safety:
- Always validate suggestions out-of-sample.
- Keep a changelog of applied modifications.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable
import os
import time

try:
    # Optional import; user must install openai
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - absence handled gracefully
    OpenAI = None  # sentinel
import pandas as pd

SUGGESTION_SCHEMA_EXAMPLE = [
    {
        "change": "Relax AO slope for bearish to ao >= ao.shift(1) - 0.0005",
        "rationale": "Losing trades show many near-flat AO slopes rejected despite good structure.",
        "implementation_hint": "In divergent_bar_indicator replace (ao > ao.shift(1)) with (ao >= ao.shift(1) - 0.0005)"
    }
]

@dataclass
class TradeContext:
    direction: str
    entry_bar: int
    exit_bar: int | None
    pnl: float
    return_pct: float
    duration: int
    entry_signal_val: float
    exit_signal_val: float

@dataclass
class PerformanceSlice:
    total_trades: int
    win_rate: float
    expectancy_pct: float
    max_dd_pct: float
    sharpe: float

@dataclass
class StrategyContext:
    performance: PerformanceSlice
    top_winners: List[TradeContext]
    top_losers: List[TradeContext]

PROMPT_TEMPLATE = """
You are a systematic trading strategy reviewer.
Given JSON context below, propose up to 3 precise rule modifications that could improve risk-adjusted performance without materially increasing trade count.
Return STRICT JSON list (no commentary) with objects having keys: change, rationale, implementation_hint.
Focus on modifying existing conditions (AO slope, structural filters, trailing logic) or adding reject-filters (volatility, time, regime) using only data available at decision time.
JSON Context:
{context_json}
Schema example for your response:
{schema_example}
""".strip()


def extract_context(stats: Dict[str, Any], max_trades: int = 30) -> StrategyContext:
    trades: pd.DataFrame = stats['_trades']
    perf = PerformanceSlice(
        total_trades=int(stats['# Trades']),
        win_rate=float(stats['Win Rate [%]']),
        expectancy_pct=float(stats['Expectancy [%]']),
        max_dd_pct=float(stats['Max. Drawdown [%]']),
        sharpe=float(stats['Sharpe Ratio'])
    )
    # Sort by return pct
    sorted_trades = trades.assign(ReturnPct=trades['ReturnPct']).sort_values('ReturnPct')
    losers_df = sorted_trades.head(min(5, len(sorted_trades)))
    winners_df = sorted_trades.tail(min(5, len(sorted_trades)))

    def row_to_ctx(row: pd.Series) -> TradeContext:
        size = row['Size']
        direction = 'long' if size > 0 else 'short'
        return TradeContext(
            direction=direction,
            entry_bar=int(row['EntryBar']),
            exit_bar=int(row['ExitBar']) if not pd.isna(row['ExitBar']) else None,
            pnl=float(row['PnL']),
            return_pct=float(row['ReturnPct']),
            duration=int(row['ExitBar'] - row['EntryBar']) if not pd.isna(row['ExitBar']) else 0,
            entry_signal_val=float(row.get('Entry_DivergentBar', 0)),
            exit_signal_val=float(row.get('Exit_DivergentBar', 0)),
        )

    losers = [row_to_ctx(r) for _, r in losers_df.iterrows()]
    winners = [row_to_ctx(r) for _, r in winners_df.iterrows()]

    return StrategyContext(performance=perf, top_winners=winners, top_losers=losers)


def build_condition_diagnostics(strat: Any) -> List[Dict[str, Any]]:
    """Summarize per-condition pass rates & win rates if condition matrix exists.

    Expects strat._cond_matrix (DataFrame) & strat._broker (Backtesting internal) to inspect trades.
    This function is resilient: returns [] if data unavailable.
    """
    import pandas as pd
    try:
        cm: pd.DataFrame = strat._strategy._cond_matrix  # type: ignore
    except Exception:
        return []
    if 'signal' not in cm.columns:
        return []
    diag = []
    for col in cm.columns:
        if col == 'signal':
            continue
        series = cm[col]
        pass_rate = float(series.mean())
        # Correlate with eventual signal occurrence
        sig_when_pass = cm.loc[series == 1, 'signal']
        signal_rate = float((sig_when_pass != 0).mean()) if len(sig_when_pass) else 0.0
        diag.append({
            'condition': col,
            'pass_rate': pass_rate,
            'signal_rate_given_pass': signal_rate,
        })
    return diag


def build_llm_prompt(ctx: StrategyContext) -> str:
    context_json = json.dumps({
        'performance': asdict(ctx.performance),
        'top_winners': [asdict(t) for t in ctx.top_winners],
        'top_losers': [asdict(t) for t in ctx.top_losers],
    }, indent=2)
    return PROMPT_TEMPLATE.format(context_json=context_json, schema_example=json.dumps(SUGGESTION_SCHEMA_EXAMPLE, indent=2))


def parse_suggestions(raw_text: str) -> List[Dict[str, str]]:
    """Attempt to parse LLM JSON suggestions; returns empty list on failure."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            cleaned = []
            for item in data:
                if not isinstance(item, dict):
                    continue
                change = str(item.get('change', '')).strip()
                rationale = str(item.get('rationale', '')).strip()
                hint = str(item.get('implementation_hint', '')).strip()
                if change:
                    cleaned.append({'change': change, 'rationale': rationale, 'implementation_hint': hint})
            return cleaned
    except Exception:
        return []
    return []


# ----------------------------- LLM Interface ----------------------------- #
def send_prompt_openai(prompt: str, model: str = "gpt-4o-mini", max_retries: int = 3, timeout_s: float = 30.0) -> Optional[str]:
    """Send prompt to OpenAI Chat API and return raw content string.

    Reads API key from environment variable OPENAI_API_KEY.
    Safe guards:
      - Returns None if openai lib not installed or key missing.
      - Simple exponential backoff on retry-able errors.
    """
    if OpenAI is None:
        print("[LLM] openai package not installed. Run: pip install openai")
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[LLM] OPENAI_API_KEY not set in environment.")
        return None
    client = OpenAI(api_key=api_key)
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a quantitative trading strategy assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=800,
            )
            elapsed = time.time() - start
            content = resp.choices[0].message.content if resp.choices else ""
            print(f"[LLM] Received response in {elapsed:.2f}s, length={len(content)}")
            return content
        except Exception as e:  # pragma: no cover
            print(f"[LLM] Error attempt {attempt}: {e}")
            if attempt == max_retries:
                return None
            time.sleep(delay)
            delay *= 1.8
    return None


def generate_rule_suggestions(stats: Dict[str, Any], model: str = "gpt-4o-mini") -> List[Dict[str, str]]:
    """High-level helper: extract context -> build prompt -> query OpenAI -> parse suggestions."""
    ctx = extract_context(stats)
    prompt = build_llm_prompt(ctx)
    raw = send_prompt_openai(prompt, model=model)
    if raw is None:
        return []
    suggestions = parse_suggestions(raw)
    return suggestions


# Placeholder for future automatic patch application.
# def apply_suggestion(...):
#     pass
