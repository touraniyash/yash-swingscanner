import time
import pandas as pd
import numpy as np
import yfinance as yf

START_DATE = "2021-01-01"
END_DATE = "2026-08-01"

EMA_FAST = 20
EMA_MID = 50
EMA_SLOW = 200

RSI_MIN = 50
RSI_MAX = 68

VOLUME_MULTIPLIER = 1.5
MIN_SCORE = 90
TARGET_R = 2.5
MAX_HOLD_DAYS = 15


def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def get_stocks():
    url = (
        "https://archives.nseindia.com/content/indices/"
        "ind_nifty500list.csv"
    )

    try:
        df = pd.read_csv(url)

        symbols = (
            df["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        return [s + ".NS" for s in symbols]

    except Exception as e:
        print("Could not download stock list:", e)
        return []


def score_stock(data):

    if len(data) < 220:
        return None

    close = data["Close"]
    volume = data["Volume"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    rsi = calculate_rsi(close)

    avg_volume = volume.rolling(20).mean()

    price = float(close.iloc[-1])

    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    e200 = float(ema200.iloc[-1])

    current_rsi = float(rsi.iloc[-1])

    if np.isnan(current_rsi):
        return None

    avg_vol = float(avg_volume.iloc[-1])

    if avg_vol <= 0:
        return None

    volume_ratio = float(volume.iloc[-1]) / avg_vol

    score = 0

    # EMA TREND - 25
    if price > e20:
        score += 8

    if e20 > e50:
        score += 8

    if e50 > e200:
        score += 9

    # MOMENTUM - 15
    if price > float(close.iloc[-6]):
        score += 7

    if price > float(close.iloc[-21]):
        score += 8

    # RSI - 10
    if 50 <= current_rsi <= 68:
        score += 10
    elif 45 <= current_rsi < 50 or 68 < current_rsi <= 72:
        score += 5

    # VOLUME - 10
    if volume_ratio >= 1.5:
        score += 10
    elif volume_ratio >= 1.2:
        score += 5

    # BREAKOUT / PULLBACK - 10
    recent_high = float(close.iloc[-21:-1].max())

    if price >= recent_high:
        score += 10
    elif price >= e20:
        score += 5

    # ENTRY / STOP
    entry = price

    recent_low = float(data["Low"].iloc[-11:].min())

    stop = recent_low

    risk = entry - stop

    if risk <= 0:
        return None

    risk_percent = risk / entry

    if risk_percent > 0.12:
        return None

    target = entry + (risk * TARGET_R)

    rr = (target - entry) / risk

    # R:R - 10
    if rr >= 2.5:
        score += 10
    elif rr >= 2.0:
        score += 7
    elif rr >= 1.5:
        score += 4

    return {
        "score": score,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rsi": current_rsi,
        "volume_ratio": volume_ratio,
        "risk_percent": risk_percent * 100,
    }


def test_trade(data, entry_index, entry, stop, target):

    future = data.iloc[
        entry_index + 1:
        entry_index + 1 + MAX_HOLD_DAYS
    ]

    if len(future) == 0:
        return "NO_DATA", None, 0

    for i, row in future.iterrows():

        high = float(row["High"])
        low = float(row["Low"])

        if low <= stop:
            return "LOSS", i, -1.0

        if high >= target:
            return "WIN", i, TARGET_R

    last_close = float(future["Close"].iloc[-1])

    r_multiple = (
        (last_close - entry) /
        (entry - stop)
    )

    return "TIME_EXIT", future.index[-1], r_multiple


def backtest_stock(symbol):

    print("Testing:", symbol)

    try:

        data = yf.download(
            symbol,
            start=START_DATE,
            end=END_DATE,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return []

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        for column in required:
            if column not in data.columns:
                return []

        data = data.dropna(subset=required)

        if len(data) < 250:
            return []

        trades = []

        for i in range(
            220,
            len(data) - MAX_HOLD_DAYS
        ):

            historical = data.iloc[:i + 1].copy()

            result = score_stock(historical)

            if result is None:
                continue

            if result["score"] < MIN_SCORE:
                continue

            entry = result["entry"]
            stop = result["stop"]
            target = result["target"]

            outcome, exit_date, r_multiple = test_trade(
                data,
                i,
                entry,
                stop,
                target
            )

            trades.append({

                "symbol": symbol.replace(".NS", ""),

                "entry_date": data.index[i],

                "exit_date": exit_date,

                "score": result["score"],

                "entry": round(entry, 2),

                "stop": round(stop, 2),

                "target": round(target, 2),

                "rsi": round(result["rsi"], 2),

                "volume_ratio": round(
                    result["volume_ratio"],
                    2
                ),

                "risk_percent": round(
                    result["risk_percent"],
                    2
                ),

                "outcome": outcome,

                "R": round(r_multiple, 2)
            })

    except Exception as e:

        print("ERROR:", symbol, e)

        return []

    return trades


def main():

    print("=" * 60)
    print("YASH SWING SCANNER - BACKTEST")
    print("=" * 60)

    print()
    print("Period:", START_DATE, "to", END_DATE)
    print("Minimum score:", MIN_SCORE)
    print("Target:", TARGET_R, "R")
    print("Holding period:", MAX_HOLD_DAYS, "days")
    print()

    stocks = get_stocks()

    if not stocks:
        print("No stocks found.")
        return

    print("Stocks found:", len(stocks))
    print()

    all_trades = []

    for count, symbol in enumerate(stocks, 1):

        print(
            f"[{count}/{len(stocks)}] {symbol}"
        )

        trades = backtest_stock(symbol)

        all_trades.extend(trades)

        time.sleep(0.3)

    if not all_trades:

        print()
        print("NO HISTORICAL TRADES FOUND.")
        return

    results = pd.DataFrame(all_trades)

    results.to_csv(
        "backtest_results.csv",
        index=False
    )

    total = len(results)

    wins = len(
        results[
            results["outcome"] == "WIN"
        ]
    )

    losses = len(
        results[
            results["outcome"] == "LOSS"
        ]
    )

    time_exits = len(
        results[
            results["outcome"] == "TIME_EXIT"
        ]
    )

    win_rate = (
        wins / total * 100
        if total else 0
    )

    total_R = results["R"].sum()

    average_R = results["R"].mean()

    gross_profit = results.loc[
        results["R"] > 0,
        "R"
    ].sum()

    gross_loss = abs(
        results.loc[
            results["R"] < 0,
            "R"
        ].sum()
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.inf
    )

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)

    print()
    print("Total trades:", total)
    print("Wins:", wins)
    print("Losses:", losses)
    print("Time exits:", time_exits)

    print(
        "Win rate:",
        round(win_rate, 2),
        "%"
    )

    print(
        "Total R:",
        round(total_R, 2)
    )

    print(
        "Average R/trade:",
        round(average_R, 2)
    )

    print(
        "Profit factor:",
        round(profit_factor, 2)
    )

    print()
    print("Results saved to backtest_results.csv")


if __name__ == "__main__":
    main()
