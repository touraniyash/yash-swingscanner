"""
Yash Swing Strategy - staged parameter optimizer

This is designed to improve the baseline backtest rather than simply
increase win rate. It searches for parameter combinations that have:
- positive expectancy
- profit factor above 1
- enough trades
- controlled drawdown

Data:
- Yahoo Finance daily NSE data
- Universe comes from symbols.txt

Stage 1: optimize signal filters
Stage 2: optimize exits/holding period for the best signal filters
Stage 3: optimize entry method for the best candidates

Outputs:
- optimization_results.csv
- best_strategy.json
- best_trades.csv

Important:
This is a research/backtest tool. It does not guarantee future returns.
"""

import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()
TARGET_BASE = 2.0
HOLD_BASE = 10
MIN_TRADES = 150
MAX_SYMBOLS = 500

RSI_RANGES = [
    (45, 65),
    (50, 65),
    (50, 68),
    (55, 70),
    (55, 65),
]
VOLUME_MIN = [1.2, 1.5, 2.0, 2.5]
TARGETS = [1.5, 2.0, 2.5, 3.0]
STOP_METHODS = ["SIGNAL_LOW", "ATR_1", "ATR_1_5"]
HOLDS = [5, 10, 15]
ENTRIES = ["NEXT_OPEN", "SIGNAL_HIGH_NEXT_DAY"]

# Small round-trip slippage assumption. Brokerage/taxes are not fully modeled.
SLIPPAGE_PCT_PER_SIDE = 0.05


def read_symbols():
    p = Path("symbols.txt")
    if not p.exists():
        raise FileNotFoundError("symbols.txt is missing")
    symbols = [x.strip() for x in p.read_text().splitlines() if x.strip()]
    return symbols[:MAX_SYMBOLS]


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def prepare(df):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in cols):
        return None

    df = df[cols].copy().dropna()
    if len(df) < 220:
        return None

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI"] = rsi(df["Close"])
    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolAvg20"]
    df["ATR"] = average_true_range(df, 14)
    return df.dropna(subset=["EMA200", "RSI", "VolumeRatio", "ATR"])


def average_true_range(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def download_symbol(symbol):
    try:
        df = yf.download(
            symbol,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        return prepare(df)
    except Exception as exc:
        print(f"{symbol}: download/prepare failed: {exc}")
        return None


def signal(df, i, rsi_lo, rsi_hi, volume_min):
    row = df.iloc[i]
    return bool(
        row["Close"] > row["EMA20"]
        and row["EMA20"] > row["EMA50"]
        and row["EMA50"] > row["EMA200"]
        and rsi_lo <= row["RSI"] <= rsi_hi
        and row["VolumeRatio"] >= volume_min
    )


def get_entry(df, signal_i, entry_method):
    if signal_i + 1 >= len(df):
        return None

    next_day = df.iloc[signal_i + 1]

    if entry_method == "NEXT_OPEN":
        return signal_i + 1, float(next_day["Open"])

    # Breakout entry: buy at the signal candle high if next day's range
    # reaches that price. If it gaps above, use next day's open.
    signal_high = float(df.iloc[signal_i]["High"])
    if float(next_day["High"]) >= signal_high:
        entry = min(float(next_day["Open"]), signal_high)
        return signal_i + 1, entry

    return None


def trade(df, signal_i, target_r, stop_method, hold_days, entry_method):
    entry_info = get_entry(df, signal_i, entry_method)
    if entry_info is None:
        return None

    entry_i, entry = entry_info
    signal_row = df.iloc[signal_i]

    if stop_method == "SIGNAL_LOW":
        stop = float(signal_row["Low"])
    elif stop_method == "ATR_1":
        stop = entry - float(signal_row["ATR"])
    else:
        stop = entry - 1.5 * float(signal_row["ATR"])

    risk = entry - stop
    if not np.isfinite(risk) or risk <= 0:
        return None

    target = entry + target_r * risk
    last_i = min(entry_i + hold_days - 1, len(df) - 1)

    exit_i = last_i
    exit_price = float(df.iloc[last_i]["Close"])
    reason = "TIME"

    for j in range(entry_i, last_i + 1):
        row = df.iloc[j]
        low = float(row["Low"])
        high = float(row["High"])

        # Conservative if both stop and target occur on one daily candle.
        if low <= stop:
            exit_i = j
            exit_price = stop
            reason = "STOP"
            break
        if high >= target:
            exit_i = j
            exit_price = target
            reason = "TARGET"
            break

    # Apply a simple 0.05% per-side slippage assumption.
    effective_entry = entry * (1 + SLIPPAGE_PCT_PER_SIDE / 100)
    effective_exit = exit_price * (1 - SLIPPAGE_PCT_PER_SIDE / 100)
    r_value = (effective_exit - effective_entry) / risk

    return {
        "SignalDate": df.index[signal_i].date().isoformat(),
        "EntryDate": df.index[entry_i].date().isoformat(),
        "ExitDate": df.index[exit_i].date().isoformat(),
        "Entry": round(entry, 4),
        "Stop": round(stop, 4),
        "Target": round(target, 4),
        "Exit": round(exit_price, 4),
        "R": round(r_value, 5),
        "ReturnPct": round((effective_exit / effective_entry - 1) * 100, 5),
        "HoldDays": int(exit_i - entry_i + 1),
        "ExitReason": reason,
    }


def run_candidate(data, rsi_lo, rsi_hi, vol_min, target_r, stop_method, hold_days, entry_method):
    trades = []
    for symbol, df in data.items():
        last_signal_i = -9999
        for i in range(len(df) - 1):
            if i - last_signal_i < 15:
                continue
            if not signal(df, i, rsi_lo, rsi_hi, vol_min):
                continue

            t = trade(df, i, target_r, stop_method, hold_days, entry_method)
            if t is None:
                continue

            t["Symbol"] = symbol.replace(".NS", "")
            t["RSI"] = round(float(df.iloc[i]["RSI"]), 2)
            t["VolumeRatio"] = round(float(df.iloc[i]["VolumeRatio"]), 2)
            t["TargetR"] = target_r
            t["StopMethod"] = stop_method
            t["HoldPlan"] = hold_days
            t["EntryMethod"] = entry_method
            t["RSILow"] = rsi_lo
            t["RSIHigh"] = rsi_hi
            t["VolumeMin"] = vol_min
            trades.append(t)
            last_signal_i = i
    return pd.DataFrame(trades)


def metrics(trades):
    if trades.empty:
        return None

    r = trades["R"].astype(float)
    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else math.inf
    equity = r.cumsum()
    dd = equity - equity.cummax()

    win_rate = (r > 0).mean() * 100
    expectancy = r.mean()
    total_r = r.sum()

    # Score favors profitability and PF, but penalizes deep drawdown.
    score = (
        expectancy * 100
        + min(pf, 3) * 15
        + min(win_rate, 70) * 0.10
        - abs(dd.min()) * 0.05
    )

    return {
        "Trades": len(r),
        "WinRatePct": round(win_rate, 2),
        "TotalR": round(total_r, 2),
        "AverageR": round(expectancy, 4),
        "ProfitFactor": round(pf, 4) if math.isfinite(pf) else 999,
        "MaxDrawdownR": round(float(dd.min()), 2),
        "Score": round(score, 4),
    }


def record_result(rows, params, m):
    if m is None:
        return
    if m["Trades"] < MIN_TRADES:
        return
    row = dict(params)
    row.update(m)
    rows.append(row)


def main():
    print("=" * 70)
    print("YASH SWING STRATEGY - PARAMETER OPTIMIZER")
    print("=" * 70)
    print(f"Data: {START_DATE} to {END_DATE}")
    print(f"Minimum trades per candidate: {MIN_TRADES}")
    print(f"Slippage: {SLIPPAGE_PCT_PER_SIDE}% per side")
    print()

    symbols = read_symbols()
    data = {}

    for n, symbol in enumerate(symbols, 1):
        print(f"[{n}/{len(symbols)}] Downloading {symbol}")
        df = download_symbol(symbol)
        if df is not None:
            data[symbol] = df
        time.sleep(0.03)

    if not data:
        raise RuntimeError("No market data was downloaded.")

    print()
    print(f"Usable symbols: {len(data)}")
    print()

    # ---------------- Stage 1: signal filters ----------------
    stage1 = []
    for rlo, rhi in RSI_RANGES:
        for vol in VOLUME_MIN:
            params = {
                "RSILow": rlo,
                "RSIHigh": rhi,
                "VolumeMin": vol,
                "TargetR": TARGET_BASE,
                "StopMethod": "SIGNAL_LOW",
                "HoldPlan": HOLD_BASE,
                "EntryMethod": "NEXT_OPEN",
                "Stage": "SIGNAL",
            }
            t = run_candidate(data, rlo, rhi, vol, TARGET_BASE,
                              "SIGNAL_LOW", HOLD_BASE, "NEXT_OPEN")
            m = metrics(t)
            record_result(stage1, params, m)

    stage1_df = pd.DataFrame(stage1).sort_values(
        ["Score", "ProfitFactor", "AverageR"], ascending=False
    )

    if stage1_df.empty:
        raise RuntimeError("Stage 1 produced no candidate with enough trades.")

    top_signal = stage1_df.head(5)

    # ---------------- Stage 2: exits ----------------
    stage2 = []
    for _, s in top_signal.iterrows():
        rlo = float(s["RSILow"])
        rhi = float(s["RSIHigh"])
        vol = float(s["VolumeMin"])

        for target in TARGETS:
            for stop in STOP_METHODS:
                for hold in HOLDS:
                    params = {
                        "RSILow": rlo,
                        "RSIHigh": rhi,
                        "VolumeMin": vol,
                        "TargetR": target,
                        "StopMethod": stop,
                        "HoldPlan": hold,
                        "EntryMethod": "NEXT_OPEN",
                        "Stage": "EXIT",
                    }
                    t = run_candidate(
                        data, rlo, rhi, vol, target, stop, hold, "NEXT_OPEN"
                    )
                    m = metrics(t)
                    record_result(stage2, params, m)

    stage2_df = pd.DataFrame(stage2)
    if not stage2_df.empty:
        stage2_df = stage2_df.sort_values(
            ["Score", "ProfitFactor", "AverageR"], ascending=False
        )

    # ---------------- Stage 3: entry ----------------
    stage3 = []
    finalists = stage2_df.head(10) if not stage2_df.empty else top_signal.head(5)

    for _, s in finalists.iterrows():
        for entry in ENTRIES:
            rlo = float(s["RSILow"])
            rhi = float(s["RSIHigh"])
            vol = float(s["VolumeMin"])
            target = float(s["TargetR"])
            stop = str(s["StopMethod"])
            hold = int(s["HoldPlan"])

            params = {
                "RSILow": rlo,
                "RSIHigh": rhi,
                "VolumeMin": vol,
                "TargetR": target,
                "StopMethod": stop,
                "HoldPlan": hold,
                "EntryMethod": entry,
                "Stage": "ENTRY",
            }
            t = run_candidate(
                data, rlo, rhi, vol, target, stop, hold, entry
            )
            m = metrics(t)
            record_result(stage3, params, m)

    stage3_df = pd.DataFrame(stage3)

    all_results = pd.concat(
        [stage1_df, stage2_df, stage3_df],
        ignore_index=True
    )

    if all_results.empty:
        raise RuntimeError("No valid optimization results.")

    all_results = all_results.sort_values(
        ["Score", "ProfitFactor", "AverageR"], ascending=False
    )
    all_results.to_csv("optimization_results.csv", index=False)

    best = all_results.iloc[0].to_dict()
    with open("best_strategy.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    # Re-run the winner and save its trade-level results.
    t = run_candidate(
        data,
        float(best["RSILow"]),
        float(best["RSIHigh"]),
        float(best["VolumeMin"]),
        float(best["TargetR"]),
        str(best["StopMethod"]),
        int(best["HoldPlan"]),
        str(best["EntryMethod"]),
    )
    t.to_csv("best_trades.csv", index=False)

    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(json.dumps(best, indent=2))
    print()
    print("Files created:")
    print("  optimization_results.csv")
    print("  best_strategy.json")
    print("  best_trades.csv")


if __name__ == "__main__":
    main()
