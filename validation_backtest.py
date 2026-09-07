import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# YASH SWING STRATEGY - FIXED PARAMETER VALIDATION
# ============================================================
# IMPORTANT:
# The parameters below are the current optimizer winner.
# The current winner was originally selected using the full
# 2020-to-current sample, so the 2025+ section is a diagnostic
# holdout check, NOT a pristine out-of-sample test.
#
# This script:
#   1) downloads market data once
#   2) runs the locked strategy on 2020-2024
#   3) runs the same locked strategy on 2025-current
#   4) runs nearby parameter combinations as a robustness check
#   5) calculates drawdown AFTER sorting trades chronologically
#   6) reports yearly/monthly and exit-reason statistics
# ============================================================

DATA_START = "2019-01-01"  # warm-up for EMA200/indicators
DATA_END = date.today().isoformat()

TRAIN_START = "2020-01-01"
TRAIN_END = "2024-12-31"

TEST_START = "2025-01-01"
TEST_END = DATA_END

MAX_SYMBOLS = 500
BATCH_SIZE = 50
MIN_ROWS = 220

SLIPPAGE_PCT_PER_SIDE = 0.05

# LOCKED WINNER FROM CURRENT OPTIMIZER
BASE_PARAMS = {
    "RSILow": 55.0,
    "RSIHigh": 65.0,
    "VolumeMin": 2.5,
    "TargetR": 3.0,
    "StopMethod": "ATR_1_5",
    "HoldPlan": 15,
    "EntryMethod": "SIGNAL_HIGH_NEXT_DAY",
}

# Robustness test only. These are NOT used to pick a new winner.
ROBUSTNESS = [
    {"Name": "BASE", "RSILow": 55.0, "RSIHigh": 65.0, "VolumeMin": 2.5, "TargetR": 3.0},
    {"Name": "RSI_50_65", "RSILow": 50.0, "RSIHigh": 65.0, "VolumeMin": 2.5, "TargetR": 3.0},
    {"Name": "RSI_45_65", "RSILow": 45.0, "RSIHigh": 65.0, "VolumeMin": 2.5, "TargetR": 3.0},
    {"Name": "VOL_2_0", "RSILow": 55.0, "RSIHigh": 65.0, "VolumeMin": 2.0, "TargetR": 3.0},
    {"Name": "TARGET_2_5", "RSILow": 55.0, "RSIHigh": 65.0, "VolumeMin": 2.5, "TargetR": 2.5},
    {"Name": "TARGET_2_0", "RSILow": 55.0, "RSIHigh": 65.0, "VolumeMin": 2.5, "TargetR": 2.0},
]


def read_symbols():
    path = Path("symbols.txt")
    if not path.exists():
        raise FileNotFoundError(
            "symbols.txt is missing. Keep the same symbols.txt used by the scanner."
        )

    symbols = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    symbols = symbols[:MAX_SYMBOLS]

    clean = []
    for symbol in symbols:
        symbol = symbol.upper()
        if not symbol.endswith(".NS"):
            symbol += ".NS"
        clean.append(symbol)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(clean))


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return (100 - 100 / (1 + rs)).fillna(50)


def average_true_range(df, period=14):
    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def prepare(df):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        # This function expects a single-ticker frame.
        if len(df.columns.levels) > 1:
            try:
                df.columns = df.columns.get_level_values(-1)
            except Exception:
                df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required):
        return None

    df = df[required].copy()
    df = df.dropna()

    if len(df) < MIN_ROWS:
        return None

    df["EMA20"] = df["Close"].ewm(
        span=20, adjust=False
    ).mean()

    df["EMA50"] = df["Close"].ewm(
        span=50, adjust=False
    ).mean()

    df["EMA200"] = df["Close"].ewm(
        span=200, adjust=False
    ).mean()

    df["RSI"] = rsi(df["Close"])

    df["VolAvg20"] = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolAvg20"]

    df["ATR"] = average_true_range(df, 14)

    df = df.dropna(
        subset=["EMA200", "RSI", "VolumeRatio", "ATR"]
    )

    return df


def download_single(symbol):
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                symbol,
                start=DATA_START,
                end=DATA_END,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="column",
                timeout=20,
            )

            df = prepare(raw)

            if df is not None:
                return df

            return None

        except Exception as exc:
            print(
                f"  {symbol}: attempt {attempt}/3 failed: {exc}",
                flush=True,
            )
            if attempt < 3:
                time.sleep(2 * attempt)

    return None


def download_batch(symbols):
    result = {}

    if not symbols:
        return result

    print(
        f"Downloading batch {len(symbols)} stocks...",
        flush=True,
    )

    try:
        raw = yf.download(
            symbols,
            start=DATA_START,
            end=DATA_END,
            auto_adjust=False,
            progress=False,
            threads=True,
            group_by="ticker",
            timeout=30,
        )
    except Exception as exc:
        print(
            f"Batch failed: {exc}",
            flush=True,
        )
        raw = None

    if raw is not None and not raw.empty:
        if len(symbols) == 1:
            df = prepare(raw)
            if df is not None:
                result[symbols[0]] = df
        elif isinstance(raw.columns, pd.MultiIndex):
            for symbol in symbols:
                try:
                    if symbol not in raw.columns.get_level_values(0):
                        continue

                    df = prepare(raw[symbol])

                    if df is not None:
                        result[symbol] = df

                except Exception as exc:
                    print(
                        f"  {symbol}: batch parse failed: {exc}",
                        flush=True,
                    )

    # Retry missing symbols individually.
    missing = [s for s in symbols if s not in result]

    if missing:
        print(
            f"Retrying {len(missing)} missing symbols individually...",
            flush=True,
        )

        for symbol in missing:
            df = download_single(symbol)
            if df is not None:
                result[symbol] = df
                print(
                    f"  OK {symbol}: {len(df)} rows",
                    flush=True,
                )
            else:
                print(
                    f"  SKIP {symbol}",
                    flush=True,
                )

    return result


def signal(df, i, params):
    row = df.iloc[i]

    return bool(
        row["Close"] > row["EMA20"]
        and row["EMA20"] > row["EMA50"]
        and row["EMA50"] > row["EMA200"]
        and params["RSILow"] <= row["RSI"] <= params["RSIHigh"]
        and row["VolumeRatio"] >= params["VolumeMin"]
    )


def get_entry(df, signal_i, entry_method):
    if signal_i + 1 >= len(df):
        return None

    next_day = df.iloc[signal_i + 1]

    if entry_method == "NEXT_OPEN":
        return signal_i + 1, float(next_day["Open"])

    signal_high = float(df.iloc[signal_i]["High"])

    if float(next_day["High"]) >= signal_high:
        # Kept identical to the existing strategy for comparability.
        entry = min(
            float(next_day["Open"]),
            signal_high,
        )
        return signal_i + 1, entry

    return None


def make_trade(df, signal_i, params):
    entry_info = get_entry(
        df,
        signal_i,
        params["EntryMethod"],
    )

    if entry_info is None:
        return None

    entry_i, entry = entry_info
    signal_row = df.iloc[signal_i]

    if params["StopMethod"] == "SIGNAL_LOW":
        stop = float(signal_row["Low"])
    elif params["StopMethod"] == "ATR_1":
        stop = entry - float(signal_row["ATR"])
    else:
        stop = entry - 1.5 * float(signal_row["ATR"])

    risk = entry - stop

    if not np.isfinite(risk) or risk <= 0:
        return None

    target = entry + params["TargetR"] * risk

    last_i = min(
        entry_i + int(params["HoldPlan"]) - 1,
        len(df) - 1,
    )

    exit_i = last_i
    exit_price = float(df.iloc[last_i]["Close"])
    reason = "TIME"

    for j in range(entry_i, last_i + 1):
        row = df.iloc[j]
        low = float(row["Low"])
        high = float(row["High"])

        # Same conservative ordering as the existing backtest:
        # stop is checked before target when both occur in one bar.
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

    effective_entry = entry * (
        1 + SLIPPAGE_PCT_PER_SIDE / 100
    )
    effective_exit = exit_price * (
        1 - SLIPPAGE_PCT_PER_SIDE / 100
    )

    r_value = (
        effective_exit - effective_entry
    ) / risk

    return {
        "SignalDate": df.index[signal_i].date().isoformat(),
        "EntryDate": df.index[entry_i].date().isoformat(),
        "ExitDate": df.index[exit_i].date().isoformat(),
        "Entry": round(entry, 4),
        "Stop": round(stop, 4),
        "Target": round(target, 4),
        "Exit": round(exit_price, 4),
        "R": round(float(r_value), 5),
        "ReturnPct": round(
            (
                effective_exit / effective_entry - 1
            ) * 100,
            5,
        ),
        "HoldDays": int(exit_i - entry_i + 1),
        "ExitReason": reason,
    }


def build_trades(data, params, start_date, end_date):
    trades = []

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for symbol, df in data.items():
        last_signal_i = -999999

        for i in range(len(df) - 1):
            signal_date = pd.Timestamp(df.index[i])

            if signal_date < start_ts or signal_date > end_ts:
                continue

            # Preserve the existing strategy's 15-day signal spacing.
            if i - last_signal_i < 15:
                continue

            if not signal(df, i, params):
                continue

            trade = make_trade(df, i, params)

            if trade is None:
                continue

            trade["Symbol"] = symbol.replace(".NS", "")
            trade["RSI"] = round(float(df.iloc[i]["RSI"]), 2)
            trade["VolumeRatio"] = round(
                float(df.iloc[i]["VolumeRatio"]),
                2,
            )
            trade["TargetR"] = params["TargetR"]
            trade["StopMethod"] = params["StopMethod"]
            trade["HoldPlan"] = params["HoldPlan"]
            trade["EntryMethod"] = params["EntryMethod"]
            trade["RSILow"] = params["RSILow"]
            trade["RSIHigh"] = params["RSIHigh"]
            trade["VolumeMin"] = params["VolumeMin"]

            trades.append(trade)
            last_signal_i = i

    if not trades:
        return pd.DataFrame()

    trades = pd.DataFrame(trades)

    # CRITICAL FIX:
    # Always sort by actual exit chronology before calculating
    # trade-sequence equity/drawdown. The old optimizer accumulated
    # R values in symbol-loop order, which is not chronological.
    trades["SignalDate"] = pd.to_datetime(trades["SignalDate"])
    trades["EntryDate"] = pd.to_datetime(trades["EntryDate"])
    trades["ExitDate"] = pd.to_datetime(trades["ExitDate"])

    trades = trades.sort_values(
        ["ExitDate", "EntryDate", "Symbol", "SignalDate"]
    ).reset_index(drop=True)

    trades["TradeNumber"] = np.arange(1, len(trades) + 1)

    trades["EquityR"] = trades["R"].astype(float).cumsum()
    trades["PeakEquityR"] = trades["EquityR"].cummax()
    trades["DrawdownR"] = (
        trades["EquityR"] - trades["PeakEquityR"]
    )

    return trades


def max_consecutive_losses(r_values):
    streak = 0
    maximum = 0

    for value in r_values:
        if value < 0:
            streak += 1
            maximum = max(maximum, streak)
        else:
            streak = 0

    return maximum


def metrics(trades):
    if trades.empty:
        return {
            "Trades": 0,
            "WinRatePct": 0.0,
            "TotalR": 0.0,
            "AverageR": 0.0,
            "ProfitFactor": 0.0,
            "MaxDrawdownR": 0.0,
            "MaxConsecutiveLosses": 0,
            "TargetExits": 0,
            "StopExits": 0,
            "TimeExits": 0,
        }

    r = trades["R"].astype(float)

    wins = r[r > 0]
    losses = r[r < 0]

    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))

    pf = (
        gross_profit / gross_loss
        if gross_loss > 0
        else math.inf
    )

    return {
        "Trades": int(len(r)),
        "WinRatePct": round(float((r > 0).mean() * 100), 2),
        "TotalR": round(float(r.sum()), 2),
        "AverageR": round(float(r.mean()), 4),
        "ProfitFactor": (
            round(float(pf), 4)
            if math.isfinite(pf)
            else 999.0
        ),
        "MaxDrawdownR": round(
            float(trades["DrawdownR"].min()),
            2,
        ),
        "MaxConsecutiveLosses": int(
            max_consecutive_losses(r.tolist())
        ),
        "TargetExits": int(
            (trades["ExitReason"] == "TARGET").sum()
        ),
        "StopExits": int(
            (trades["ExitReason"] == "STOP").sum()
        ),
        "TimeExits": int(
            (trades["ExitReason"] == "TIME").sum()
        ),
    }


def add_period_metrics(trades, period_name):
    m = metrics(trades)
    m["Period"] = period_name
    return m


def yearly_report(trades):
    if trades.empty:
        return pd.DataFrame()

    tmp = trades.copy()
    tmp["Year"] = tmp["ExitDate"].dt.year

    rows = []

    for year, group in tmp.groupby("Year"):
        m = metrics(
            group.sort_values(
                ["ExitDate", "EntryDate", "Symbol"]
            )
        )
        m["Year"] = int(year)
        rows.append(m)

    return pd.DataFrame(rows).sort_values("Year")


def monthly_report(trades):
    if trades.empty:
        return pd.DataFrame()

    tmp = trades.copy()
    tmp["Month"] = tmp["ExitDate"].dt.to_period("M").astype(str)

    rows = []

    for month, group in tmp.groupby("Month"):
        m = metrics(
            group.sort_values(
                ["ExitDate", "EntryDate", "Symbol"]
            )
        )
        m["Month"] = month
        rows.append(m)

    return pd.DataFrame(rows).sort_values("Month")


def symbol_report(trades):
    if trades.empty:
        return pd.DataFrame()

    rows = []

    for symbol, group in trades.groupby("Symbol"):
        m = metrics(
            group.sort_values(
                ["ExitDate", "EntryDate"]
            )
        )
        m["Symbol"] = symbol
        rows.append(m)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["TotalR", "ProfitFactor"],
            ascending=False,
        )
    )


def robustness_report(data, period_start, period_end, period_name):
    rows = []

    for item in ROBUSTNESS:
        params = dict(BASE_PARAMS)
        params.update(
            {
                "RSILow": item["RSILow"],
                "RSIHigh": item["RSIHigh"],
                "VolumeMin": item["VolumeMin"],
                "TargetR": item["TargetR"],
            }
        )

        trades = build_trades(
            data,
            params,
            period_start,
            period_end,
        )

        m = metrics(trades)
        m["Variant"] = item["Name"]
        m["Period"] = period_name

        rows.append(m)

    return pd.DataFrame(rows)


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8",
    )


def main():
    print("=" * 78)
    print("YASH SWING STRATEGY - VALIDATION BACKTEST")
    print("=" * 78)
    print(f"Download data: {DATA_START} to {DATA_END}")
    print(f"Training period: {TRAIN_START} to {TRAIN_END}")
    print(f"Diagnostic holdout: {TEST_START} to {TEST_END}")
    print()
    print("LOCKED PARAMETERS:")
    for key, value in BASE_PARAMS.items():
        print(f"  {key}: {value}")
    print()
    print(
        "NOTE: 2025+ is diagnostic only because the current "
        "winner was originally selected using the full sample."
    )
    print()

    symbols = read_symbols()

    print(f"Symbols requested: {len(symbols)}", flush=True)

    data = {}

    for start in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[start:start + BATCH_SIZE]

        batch_data = download_batch(batch)
        data.update(batch_data)

        print(
            f"Progress: {min(start + BATCH_SIZE, len(symbols))}/"
            f"{len(symbols)} | usable: {len(data)}",
            flush=True,
        )

    if not data:
        raise RuntimeError("No market data downloaded.")

    failed = [s for s in symbols if s not in data]

    Path("validation_failed_symbols.txt").write_text(
        "\n".join(failed),
        encoding="utf-8",
    )

    print()
    print(f"Usable symbols: {len(data)}")
    print(f"Failed/unusable symbols: {len(failed)}")
    print()

    # ------------------------------------------------------------
    # MAIN VALIDATION
    # ------------------------------------------------------------
    print("Running TRAINING period...", flush=True)

    train_trades = build_trades(
        data,
        BASE_PARAMS,
        TRAIN_START,
        TRAIN_END,
    )

    print(
        f"Training trades: {len(train_trades)}",
        flush=True,
    )

    print("Running DIAGNOSTIC HOLDOUT period...", flush=True)

    test_trades = build_trades(
        data,
        BASE_PARAMS,
        TEST_START,
        TEST_END,
    )

    print(
        f"Holdout trades: {len(test_trades)}",
        flush=True,
    )

    all_trades = pd.concat(
        [train_trades, test_trades],
        ignore_index=True,
    )

    if not all_trades.empty:
        all_trades = (
            all_trades
            .sort_values(
                ["ExitDate", "EntryDate", "Symbol", "SignalDate"]
            )
            .reset_index(drop=True)
        )

        all_trades["TradeNumber"] = np.arange(
            1, len(all_trades) + 1
        )
        all_trades["EquityR"] = (
            all_trades["R"].astype(float).cumsum()
        )
        all_trades["PeakEquityR"] = (
            all_trades["EquityR"].cummax()
        )
        all_trades["DrawdownR"] = (
            all_trades["EquityR"]
            - all_trades["PeakEquityR"]
        )

    # ------------------------------------------------------------
    # SAVE TRADE FILES
    # ------------------------------------------------------------
    train_trades.to_csv(
        "validation_train_trades.csv",
        index=False,
    )
    test_trades.to_csv(
        "validation_holdout_trades.csv",
        index=False,
    )
    all_trades.to_csv(
        "validation_all_trades.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------
    summary_rows = [
        add_period_metrics(
            train_trades,
            "TRAIN_2020_2024",
        ),
        add_period_metrics(
            test_trades,
            "HOLDOUT_2025_CURRENT",
        ),
        add_period_metrics(
            all_trades,
            "ALL_2020_CURRENT",
        ),
    ]

    summary = pd.DataFrame(summary_rows)

    # Put Period first.
    cols = ["Period"] + [
        c for c in summary.columns if c != "Period"
    ]
    summary = summary[cols]

    summary.to_csv(
        "validation_summary.csv",
        index=False,
    )

    yearly = yearly_report(all_trades)
    yearly.to_csv(
        "validation_yearly.csv",
        index=False,
    )

    monthly = monthly_report(all_trades)
    monthly.to_csv(
        "validation_monthly.csv",
        index=False,
    )

    symbols_report = symbol_report(all_trades)
    symbols_report.to_csv(
        "validation_by_symbol.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # ROBUSTNESS
    # ------------------------------------------------------------
    robustness_train = robustness_report(
        data,
        TRAIN_START,
        TRAIN_END,
        "TRAIN_2020_2024",
    )

    robustness_test = robustness_report(
        data,
        TEST_START,
        TEST_END,
        "HOLDOUT_2025_CURRENT",
    )

    robustness = pd.concat(
        [robustness_train, robustness_test],
        ignore_index=True,
    )

    robustness.to_csv(
        "validation_robustness.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # CONFIG + REPORT
    # ------------------------------------------------------------
    save_json(
        "validation_parameters.json",
        {
            "base_params": BASE_PARAMS,
            "data_start": DATA_START,
            "data_end": DATA_END,
            "train_start": TRAIN_START,
            "train_end": TRAIN_END,
            "holdout_start": TEST_START,
            "holdout_end": TEST_END,
            "slippage_pct_per_side": SLIPPAGE_PCT_PER_SIDE,
            "note": (
                "Holdout is diagnostic rather than pristine OOS "
                "because the original winner was selected using "
                "the full sample."
            ),
        },
    )

    report_lines = [
        "YASH SWING STRATEGY - VALIDATION REPORT",
        "=" * 78,
        "",
        "LOCKED STRATEGY",
    ]

    for key, value in BASE_PARAMS.items():
        report_lines.append(f"{key}: {value}")

    report_lines.extend(
        [
            "",
            "PERIOD RESULTS",
            "-" * 78,
        ]
    )

    for _, row in summary.iterrows():
        report_lines.append(
            f"{row['Period']}: "
            f"Trades={int(row['Trades'])}, "
            f"WinRate={row['WinRatePct']}%, "
            f"TotalR={row['TotalR']}, "
            f"AvgR={row['AverageR']}, "
            f"PF={row['ProfitFactor']}, "
            f"MaxDD={row['MaxDrawdownR']}R, "
            f"MaxLossStreak={int(row['MaxConsecutiveLosses'])}"
        )

    report_lines.extend(
        [
            "",
            "DRAWdown methodology",
            "-" * 78,
            "Trades are sorted chronologically by ExitDate before "
            "trade-sequence equity/drawdown is calculated.",
            "This fixes the old optimizer issue where R values were "
            "accumulated in symbol-loop order.",
            "",
            "IMPORTANT ENTRY CAVEAT",
            "-" * 78,
            "SIGNAL_HIGH_NEXT_DAY is kept identical to the existing "
            "strategy for comparability. The existing implementation "
            "uses min(next_open, signal_high) after confirming that "
            "next-day high reaches signal high. This should be reviewed "
            "before live trading because it can be more favorable than "
            "a literal breakout execution.",
            "",
            "IMPORTANT OOS CAVEAT",
            "-" * 78,
            "The current winning parameters were selected using the full "
            "historical sample, including 2025+. Therefore the 2025+ "
            "section is a diagnostic holdout check, not a fully untouched "
            "out-of-sample test.",
            "",
            "FILES",
            "-" * 78,
            "validation_summary.csv",
            "validation_train_trades.csv",
            "validation_holdout_trades.csv",
            "validation_all_trades.csv",
            "validation_yearly.csv",
            "validation_monthly.csv",
            "validation_by_symbol.csv",
            "validation_robustness.csv",
            "validation_parameters.json",
            "validation_failed_symbols.txt",
        ]
    )

    Path("VALIDATION_REPORT.txt").write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("VALIDATION COMPLETE")
    print("=" * 78)
    print()
    print(summary.to_string(index=False))
    print()
    print("Files created:")
    print("  validation_summary.csv")
    print("  validation_train_trades.csv")
    print("  validation_holdout_trades.csv")
    print("  validation_all_trades.csv")
    print("  validation_yearly.csv")
    print("  validation_monthly.csv")
    print("  validation_by_symbol.csv")
    print("  validation_robustness.csv")
    print("  validation_parameters.json")
    print("  validation_failed_symbols.txt")
    print("  VALIDATION_REPORT.txt")


if __name__ == "__main__":
    main()
