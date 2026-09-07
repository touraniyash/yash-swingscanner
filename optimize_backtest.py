import json
import math
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# YASH SWING STRATEGY - FAST PARAMETER OPTIMIZER
# ============================================================

START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()

MAX_SYMBOLS = 500
BATCH_SIZE = 25

MIN_TRADES = 150

SLIPPAGE_PCT_PER_SIDE = 0.05

TARGET_BASE = 2.0
HOLD_BASE = 10

RSI_RANGES = [
    (45, 65),
    (50, 65),
    (50, 68),
    (55, 70),
    (55, 65),
]

VOLUME_MIN = [
    1.2,
    1.5,
    2.0,
    2.5,
]

TARGETS = [
    1.5,
    2.0,
    2.5,
    3.0,
]

STOP_METHODS = [
    "SIGNAL_LOW",
    "ATR_1",
    "ATR_1_5",
]

HOLDS = [
    5,
    10,
    15,
]

ENTRIES = [
    "NEXT_OPEN",
    "SIGNAL_HIGH_NEXT_DAY",
]


# ============================================================
# SYMBOL LIST
# ============================================================

def read_symbols():

    p = Path("symbols.txt")

    if not p.exists():
        raise FileNotFoundError(
            "symbols.txt is missing"
        )

    symbols = [
        x.strip()
        for x in p.read_text().splitlines()
        if x.strip()
    ]

    symbols = symbols[:MAX_SYMBOLS]

    # Make sure Yahoo symbols have .NS
    clean = []

    for s in symbols:

        if not s.endswith(".NS"):
            s = s + ".NS"

        clean.append(s)

    return clean


# ============================================================
# RSI
# ============================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return (
        100
        - 100 / (1 + rs)
    ).fillna(50)


# ============================================================
# ATR
# ============================================================

def average_true_range(df, period=14):

    previous_close = df["Close"].shift(1)

    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period
    ).mean()


# ============================================================
# PREPARE DATA
# ============================================================

def prepare(df):

    if df is None or df.empty:
        return None

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if any(
        c not in df.columns
        for c in required
    ):
        return None

    df = df[required].copy()

    df = df.dropna()

    if len(df) < 220:
        return None

    df["EMA20"] = (
        df["Close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    df["EMA50"] = (
        df["Close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    df["EMA200"] = (
        df["Close"]
        .ewm(
            span=200,
            adjust=False
        )
        .mean()
    )

    df["RSI"] = rsi(
        df["Close"]
    )

    df["VolAvg20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["VolumeRatio"] = (
        df["Volume"]
        / df["VolAvg20"]
    )

    df["ATR"] = average_true_range(
        df,
        14
    )

    df = df.dropna(
        subset=[
            "EMA200",
            "RSI",
            "VolumeRatio",
            "ATR",
        ]
    )

    return df


# ============================================================
# BATCH DOWNLOAD
# ============================================================

def download_batch(symbols):
    """Download symbols one at a time with timeout/retry protection.

    A single Yahoo Finance request must never block the whole optimizer.
    """
    result = {}

    for n, symbol in enumerate(symbols, 1):
        print(f"Downloading {symbol} ({n}/{len(symbols)})...", flush=True)

        success = False

        for attempt in range(1, 4):
            try:
                raw = yf.download(
                    symbol,
                    start=START_DATE,
                    end=END_DATE,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    group_by="column",
                    timeout=15,
                )

                df = prepare(raw)

                if df is not None:
                    result[symbol] = df
                    print(f"  OK {symbol}: {len(df)} rows", flush=True)
                else:
                    print(f"  SKIP {symbol}: insufficient/invalid data", flush=True)

                success = True
                break

            except Exception as exc:
                print(
                    f"  Attempt {attempt}/3 failed for {symbol}: {exc}",
                    flush=True,
                )
                if attempt < 3:
                    time.sleep(2 * attempt)

        if not success:
            print(f"  FAILED {symbol}: skipped after 3 attempts", flush=True)

    return result


# ============================================================
# SIGNAL
# ============================================================

def signal(
    df,
    i,
    rsi_lo,
    rsi_hi,
    volume_min
):

    row = df.iloc[i]

    return bool(

        row["Close"] > row["EMA20"]

        and row["EMA20"] > row["EMA50"]

        and row["EMA50"] > row["EMA200"]

        and rsi_lo <= row["RSI"] <= rsi_hi

        and row["VolumeRatio"] >= volume_min

    )


# ============================================================
# ENTRY
# ============================================================

def get_entry(
    df,
    signal_i,
    entry_method
):

    if signal_i + 1 >= len(df):
        return None

    next_day = df.iloc[
        signal_i + 1
    ]

    if entry_method == "NEXT_OPEN":

        return (
            signal_i + 1,
            float(next_day["Open"])
        )

    signal_high = float(
        df.iloc[signal_i]["High"]
    )

    if (
        float(next_day["High"])
        >= signal_high
    ):

        entry = min(
            float(next_day["Open"]),
            signal_high
        )

        return (
            signal_i + 1,
            entry
        )

    return None


# ============================================================
# TRADE
# ============================================================

def trade(
    df,
    signal_i,
    target_r,
    stop_method,
    hold_days,
    entry_method
):

    entry_info = get_entry(
        df,
        signal_i,
        entry_method
    )

    if entry_info is None:
        return None

    entry_i, entry = entry_info

    signal_row = df.iloc[
        signal_i
    ]

    if stop_method == "SIGNAL_LOW":

        stop = float(
            signal_row["Low"]
        )

    elif stop_method == "ATR_1":

        stop = (
            entry
            - float(signal_row["ATR"])
        )

    else:

        stop = (
            entry
            - 1.5
            * float(signal_row["ATR"])
        )

    risk = entry - stop

    if (
        not np.isfinite(risk)
        or risk <= 0
    ):
        return None

    target = (
        entry
        + target_r * risk
    )

    last_i = min(
        entry_i + hold_days - 1,
        len(df) - 1
    )

    exit_i = last_i

    exit_price = float(
        df.iloc[last_i]["Close"]
    )

    reason = "TIME"

    for j in range(
        entry_i,
        last_i + 1
    ):

        row = df.iloc[j]

        low = float(
            row["Low"]
        )

        high = float(
            row["High"]
        )

        # Conservative:
        # stop checked before target.
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

    effective_entry = (
        entry
        * (
            1
            + SLIPPAGE_PCT_PER_SIDE
            / 100
        )
    )

    effective_exit = (
        exit_price
        * (
            1
            - SLIPPAGE_PCT_PER_SIDE
            / 100
        )
    )

    r_value = (
        effective_exit
        - effective_entry
    ) / risk

    return {

        "SignalDate":
            df.index[signal_i]
            .date()
            .isoformat(),

        "EntryDate":
            df.index[entry_i]
            .date()
            .isoformat(),

        "ExitDate":
            df.index[exit_i]
            .date()
            .isoformat(),

        "Entry":
            round(entry, 4),

        "Stop":
            round(stop, 4),

        "Target":
            round(target, 4),

        "Exit":
            round(exit_price, 4),

        "R":
            round(r_value, 5),

        "ReturnPct":
            round(
                (
                    effective_exit
                    / effective_entry
                    - 1
                )
                * 100,
                5
            ),

        "HoldDays":
            int(
                exit_i
                - entry_i
                + 1
            ),

        "ExitReason":
            reason,
    }


# ============================================================
# RUN CANDIDATE
# ============================================================

def run_candidate(
    data,
    rsi_lo,
    rsi_hi,
    volume_min,
    target_r,
    stop_method,
    hold_days,
    entry_method
):

    trades = []

    for symbol, df in data.items():

        last_signal_i = -9999

        for i in range(
            len(df) - 1
        ):

            if (
                i - last_signal_i
                < 15
            ):
                continue

            if not signal(
                df,
                i,
                rsi_lo,
                rsi_hi,
                volume_min
            ):
                continue

            t = trade(
                df,
                i,
                target_r,
                stop_method,
                hold_days,
                entry_method
            )

            if t is None:
                continue

            t["Symbol"] = (
                symbol.replace(
                    ".NS",
                    ""
                )
            )

            t["RSI"] = round(
                float(
                    df.iloc[i]["RSI"]
                ),
                2
            )

            t["VolumeRatio"] = round(
                float(
                    df.iloc[i][
                        "VolumeRatio"
                    ]
                ),
                2
            )

            t["TargetR"] = target_r

            t["StopMethod"] = (
                stop_method
            )

            t["HoldPlan"] = (
                hold_days
            )

            t["EntryMethod"] = (
                entry_method
            )

            t["RSILow"] = rsi_lo
            t["RSIHigh"] = rsi_hi
            t["VolumeMin"] = volume_min

            trades.append(t)

            last_signal_i = i

    return pd.DataFrame(
        trades
    )


# ============================================================
# METRICS
# ============================================================

def metrics(trades):

    if trades.empty:
        return None

    r = trades["R"].astype(float)

    wins = r[r > 0]

    losses = r[r < 0]

    gross_profit = wins.sum()

    gross_loss = abs(
        losses.sum()
    )

    if gross_loss > 0:
        pf = (
            gross_profit
            / gross_loss
        )
    else:
        pf = math.inf

    equity = r.cumsum()

    drawdown = (
        equity
        - equity.cummax()
    )

    win_rate = (
        (r > 0).mean()
        * 100
    )

    expectancy = r.mean()

    total_r = r.sum()

    max_dd = float(
        drawdown.min()
    )

    score = (

        expectancy * 100

        + min(pf, 3) * 15

        + min(
            win_rate,
            70
        ) * 0.10

        - abs(max_dd)
        * 0.05

    )

    return {

        "Trades":
            len(r),

        "WinRatePct":
            round(
                win_rate,
                2
            ),

        "TotalR":
            round(
                total_r,
                2
            ),

        "AverageR":
            round(
                expectancy,
                4
            ),

        "ProfitFactor":
            round(
                pf,
                4
            )
            if math.isfinite(pf)
            else 999,

        "MaxDrawdownR":
            round(
                max_dd,
                2
            ),

        "Score":
            round(
                score,
                4
            ),
    }


# ============================================================
# RECORD RESULT
# ============================================================

def record_result(
    rows,
    params,
    metrics_result
):

    if metrics_result is None:
        return

    if (
        metrics_result["Trades"]
        < MIN_TRADES
    ):
        return

    row = dict(params)

    row.update(
        metrics_result
    )

    rows.append(row)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "YASH SWING STRATEGY"
        " - FAST PARAMETER OPTIMIZER"
    )

    print("=" * 70)

    print(
        f"Data: {START_DATE}"
        f" to {END_DATE}"
    )

    print(
        f"Minimum trades: {MIN_TRADES}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Slippage: "
        f"{SLIPPAGE_PCT_PER_SIDE}%"
        " per side"
    )

    print()

    symbols = read_symbols()

    print(
        f"Stocks to download: "
        f"{len(symbols)}"
    )

    print()

    # ========================================================
    # DOWNLOAD ALL DATA
    # ========================================================

    data = {}

    for start in range(
        0,
        len(symbols),
        BATCH_SIZE
    ):

        batch = symbols[
            start:
            start + BATCH_SIZE
        ]

        batch_data = download_batch(
            batch
        )

        data.update(
            batch_data
        )

        print(
            f"Progress: {min(start + BATCH_SIZE, len(symbols))}/{len(symbols)}"
            f" | usable: {len(data)}",
            flush=True,
        )

    if not data:

        raise RuntimeError(
            "No market data downloaded."
        )

    print()

    print(
        f"Usable symbols: "
        f"{len(data)}"
    , flush=True)

    failed_symbols = [s for s in symbols if s not in data]
    Path("failed_symbols.txt").write_text(
        "\n".join(failed_symbols),
        encoding="utf-8",
    )
    print(f"Failed/unusable symbols: {len(failed_symbols)}", flush=True)

    print()

    # ========================================================
    # STAGE 1
    # ========================================================

    print(
        "STAGE 1 - SIGNAL FILTERS"
    )

    stage1 = []

    for rlo, rhi in RSI_RANGES:

        for vol in VOLUME_MIN:

            params = {

                "RSILow": rlo,

                "RSIHigh": rhi,

                "VolumeMin": vol,

                "TargetR": TARGET_BASE,

                "StopMethod":
                    "SIGNAL_LOW",

                "HoldPlan":
                    HOLD_BASE,

                "EntryMethod":
                    "NEXT_OPEN",

                "Stage":
                    "SIGNAL",
            }

            t = run_candidate(

                data,

                rlo,
                rhi,
                vol,

                TARGET_BASE,

                "SIGNAL_LOW",

                HOLD_BASE,

                "NEXT_OPEN"

            )

            m = metrics(t)

            record_result(
                stage1,
                params,
                m
            )

    stage1_df = pd.DataFrame(
        stage1
    )

    if stage1_df.empty:

        raise RuntimeError(
            "Stage 1 produced "
            "no candidate with "
            "enough trades."
        )

    stage1_df = (
        stage1_df
        .sort_values(
            [
                "Score",
                "ProfitFactor",
                "AverageR"
            ],
            ascending=False
        )
    )

    top_signal = (
        stage1_df
        .head(5)
    )

    print(
        "Stage 1 complete.",
        flush=True
    )

    print()

    # ========================================================
    # STAGE 2
    # ========================================================

    print(
        "STAGE 2 - EXIT OPTIMIZATION"
    )

    stage2 = []

    for _, s in (
        top_signal.iterrows()
    ):

        rlo = float(
            s["RSILow"]
        )

        rhi = float(
            s["RSIHigh"]
        )

        vol = float(
            s["VolumeMin"]
        )

        for target in TARGETS:

            for stop in STOP_METHODS:

                for hold in HOLDS:

                    params = {

                        "RSILow":
                            rlo,

                        "RSIHigh":
                            rhi,

                        "VolumeMin":
                            vol,

                        "TargetR":
                            target,

                        "StopMethod":
                            stop,

                        "HoldPlan":
                            hold,

                        "EntryMethod":
                            "NEXT_OPEN",

                        "Stage":
                            "EXIT",
                    }

                    t = run_candidate(

                        data,

                        rlo,
                        rhi,
                        vol,

                        target,

                        stop,

                        hold,

                        "NEXT_OPEN"

                    )

                    m = metrics(t)

                    record_result(
                        stage2,
                        params,
                        m
                    )

    stage2_df = pd.DataFrame(
        stage2
    )

    if not stage2_df.empty:

        stage2_df = (
            stage2_df
            .sort_values(
                [
                    "Score",
                    "ProfitFactor",
                    "AverageR"
                ],
                ascending=False
            )
        )

    print(
        "Stage 2 complete.",
        flush=True
    )

    print()

    # ========================================================
    # STAGE 3
    # ========================================================

    print(
        "STAGE 3 - ENTRY OPTIMIZATION"
    )

    stage3 = []

    if not stage2_df.empty:

        finalists = (
            stage2_df
            .head(10)
        )

    else:

        finalists = (
            top_signal
            .head(5)
        )

    for _, s in (
        finalists.iterrows()
    ):

        rlo = float(
            s["RSILow"]
        )

        rhi = float(
            s["RSIHigh"]
        )

        vol = float(
            s["VolumeMin"]
        )

        target = float(
            s["TargetR"]
        )

        stop = str(
            s["StopMethod"]
        )

        hold = int(
            s["HoldPlan"]
        )

        for entry in ENTRIES:

            params = {

                "RSILow":
                    rlo,

                "RSIHigh":
                    rhi,

                "VolumeMin":
                    vol,

                "TargetR":
                    target,

                "StopMethod":
                    stop,

                "HoldPlan":
                    hold,

                "EntryMethod":
                    entry,

                "Stage":
                    "ENTRY",
            }

            t = run_candidate(

                data,

                rlo,
                rhi,
                vol,

                target,

                stop,

                hold,

                entry

            )

            m = metrics(t)

            record_result(
                stage3,
                params,
                m
            )

    stage3_df = pd.DataFrame(
        stage3
    )

    print(
        "Stage 3 complete.",
        flush=True
    )

    print()

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    frames = []

    if not stage1_df.empty:
        frames.append(
            stage1_df
        )

    if not stage2_df.empty:
        frames.append(
            stage2_df
        )

    if not stage3_df.empty:
        frames.append(
            stage3_df
        )

    all_results = pd.concat(
        frames,
        ignore_index=True
    )

    if all_results.empty:

        raise RuntimeError(
            "No valid optimization results."
        )

    all_results = (
        all_results
        .sort_values(
            [
                "Score",
                "ProfitFactor",
                "AverageR"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # Save all optimization results
    all_results.to_csv(
        "optimization_results.csv",
        index=False
    )

    # Best strategy
    best = (
        all_results
        .iloc[0]
        .to_dict()
    )

    with open(
        "best_strategy.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best,
            f,
            indent=2
        )

    # Re-run winner
    best_trades = run_candidate(

        data,

        float(
            best["RSILow"]
        ),

        float(
            best["RSIHigh"]
        ),

        float(
            best["VolumeMin"]
        ),

        float(
            best["TargetR"]
        ),

        str(
            best["StopMethod"]
        ),

        int(
            best["HoldPlan"]
        ),

        str(
            best["EntryMethod"]
        )

    )

    best_trades.to_csv(
        "best_trades.csv",
        index=False
    )

    print()

    print("=" * 70)

    print(
        "OPTIMIZATION COMPLETE",
        flush=True
    )

    print("=" * 70)

    print()

    print(
        json.dumps(
            best,
            indent=2
        )
    )

    print()

    print(
        "Files created:"
    )

    print(
        "  optimization_results.csv"
    )

    print(
        "  best_strategy.json"
    )

    print(
        "  best_trades.csv"
    )

    print()

    print(
        "Top 10 strategies:"
    )

    print()

    print(
        all_results[
            [
                "Stage",
                "RSILow",
                "RSIHigh",
                "VolumeMin",
                "TargetR",
                "StopMethod",
                "HoldPlan",
                "EntryMethod",
                "Trades",
                "WinRatePct",
                "AverageR",
                "ProfitFactor",
                "MaxDrawdownR",
                "Score",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
