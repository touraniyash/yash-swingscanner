import json, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

CONSTITUENTS = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
headers = {"User-Agent": "Mozilla/5.0"}


def get_constituents():
    r = requests.get(CONSTITUENTS, headers=headers, timeout=30)
    r.raise_for_status()

    from io import BytesIO
    df = pd.read_csv(BytesIO(r.content))

    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = next(
        c for c in df.columns
        if c.lower() == "symbol"
    )

    industry_col = next(
        (c for c in df.columns if c.lower() == "industry"),
        None
    )

    if industry_col is None:
        industry_col = "Industry"

    out = df[
        [symbol_col, industry_col]
    ].rename(
        columns={
            symbol_col: "symbol",
            industry_col: "sector"
        }
    ).dropna()

    out["symbol"] = out["symbol"].astype(str).str.strip()
    out["sector"] = out["sector"].astype(str).str.strip()

    return out.drop_duplicates("symbol")


def rsi(series, n=14):
    delta = series.diff()

    gain = delta.clip(lower=0).rolling(n).mean()
    loss = (-delta.clip(upper=0)).rolling(n).mean()

    rs = gain / loss.replace(0, np.nan)

    return (100 - 100 / (1 + rs)).fillna(50)


def calculate_atr(high, low, close, n=14):
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    return true_range.rolling(n).mean()


def score_stock(high, low, close, volume):

    # =========================
    # EMA TREND
    # =========================

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    ema50 = close.ewm(
        span=50,
        adjust=False
    ).mean()

    ema200 = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # =========================
    # RSI
    # =========================

    rrsi = rsi(close)

    # =========================
    # VOLUME
    # =========================

    volume_ratio = (
        volume /
        volume.rolling(20).mean()
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =========================
    # MOMENTUM
    # =========================

    w1 = (close / close.shift(5) - 1) * 100
    m1 = (close / close.shift(21) - 1) * 100
    m3 = (close / close.shift(63) - 1) * 100

    # =========================
    # ATR
    # =========================

    atr = calculate_atr(
        high,
        low,
        close
    )

    # =========================
    # CURRENT VALUES
    # =========================

    price = float(close.iloc[-1])

    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    e200 = float(ema200.iloc[-1])

    rsi_value = float(rrsi.iloc[-1])
    volume_value = float(volume_ratio.iloc[-1])

    week = float(w1.iloc[-1])
    month = float(m1.iloc[-1])
    three_month = float(m3.iloc[-1])

    atr_value = float(atr.iloc[-1])

    # =========================
    # EMA CONDITIONS
    # =========================

    price_above_20 = price > e20
    ema20_above_50 = e20 > e50
    ema50_above_200 = e50 > e200

    ema200_rising = (
        e200 >
        float(ema200.iloc[-21])
    )

    price_above_50 = price > e50

    # =========================
    # TREND SCORE /25
    # =========================

    trend = (
        (5 if price_above_20 else 0) +
        (5 if ema20_above_50 else 0) +
        (5 if ema50_above_200 else 0) +
        (5 if ema200_rising else 0) +
        (5 if price_above_50 else 0)
    )

    # =========================
    # MOMENTUM SCORE /15
    # =========================

    momentum = (
        (5 if week > 0 else 0) +
        (5 if month > 0 else 0) +
        (5 if three_month > 0 else 0)
    )

    # =========================
    # RSI SCORE /10
    # =========================

    if 55 <= rsi_value <= 68:
        rsi_score = 10
    elif 50 <= rsi_value < 55 or 68 < rsi_value <= 72:
        rsi_score = 7
    elif 45 <= rsi_value < 50:
        rsi_score = 3
    else:
        rsi_score = 0

    # =========================
    # VOLUME SCORE /10
    # =========================

    if volume_value >= 1.5:
        volume_score = 10
    elif volume_value >= 1.2:
        volume_score = 7
    elif volume_value >= 1.0:
        volume_score = 4
    else:
        volume_score = 0

    # =========================
    # BREAKOUT / PULLBACK
    # =========================

    previous_20_high = float(
        close.shift(1)
        .rolling(20)
        .max()
        .iloc[-1]
    )

    breakout = price > previous_20_high

    distance_from_20 = (
        abs(price - e20) / e20
        if e20
        else 1
    )

    pullback = (
        price > e20
        and distance_from_20 <= 0.04
        and float(close.iloc[-1]) >=
            float(close.iloc[-2])
    )

    setup_score = (
        10 if breakout
        else 8 if pullback
        else 0
    )

    # =========================
    # VOLATILITY / RISK SCORE
    # =========================

    atr_pct = (
        atr_value / price
        if price
        else 1
    )

    if atr_pct <= 0.025:
        risk_score = 10
    elif atr_pct <= 0.04:
        risk_score = 7
    elif atr_pct <= 0.06:
        risk_score = 4
    else:
        risk_score = 0

    return {
        "close": price,

        "ema20": e20,
        "ema50": e50,
        "ema200": e200,

        "price_above_20": price_above_20,
        "ema20_above_50": ema20_above_50,
        "ema50_above_200": ema50_above_200,
        "ema200_rising": ema200_rising,

        "rsi": rsi_value,
        "vol": volume_value,

        "w1": week,
        "m1": month,
        "m3": three_month,

        "atr": atr_value,
        "atr_pct": atr_pct,

        "trend": trend,
        "momentum": momentum,
        "rs": rsi_score,
        "volscore": volume_score,
        "setup": setup_score,
        "risk": risk_score,

        "breakout": breakout,
        "pullback": pullback
    }


def main():

    constituents = get_constituents()

    symbols = constituents.symbol.tolist()

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    stocks = []

    # =========================
    # DOWNLOAD DATA
    # =========================

    for start in range(
        0,
        len(tickers),
        80
    ):

        batch = tickers[
            start:start + 80
        ]

        try:

            raw = yf.download(
                batch,
                period="1y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False
            )

        except Exception as e:

            print(
                "batch error",
                e
            )

            continue

        for _, row in constituents.iloc[
            start:start + 80
        ].iterrows():

            symbol = row.symbol
            sector = row.sector

            try:

                data = raw[
                    symbol + ".NS"
                ].dropna()

                if len(data) < 220:
                    continue

                result = score_stock(
                    data["High"],
                    data["Low"],
                    data["Close"],
                    data["Volume"]
                )

                stocks.append({
                    "symbol": symbol,
                    "sector": sector,
                    **result
                })

            except Exception as e:

                print(
                    "skip",
                    symbol,
                    e
                )

        time.sleep(1)

    df = pd.DataFrame(stocks)

    if df.empty:
        raise RuntimeError(
            "No stock data downloaded"
        )

    # =========================
    # SECTOR STRENGTH
    # =========================

    sector_rows = []

    for sector, group in df.groupby(
        "sector"
    ):

        week = float(
            group.w1.mean()
        )

        month = float(
            group.m1.mean()
        )

        breadth = float(
            (
                group.w1.gt(0) &
                group.m1.gt(0)
            ).mean() * 100
        )

        trend_breadth = float(
            (
                group.trend >= 15
            ).mean() * 100
        )

        sector_score = (

            (5 if trend_breadth >= 50
             else 3 if trend_breadth >= 35
             else 0)

            +

            (5 if week > 0 else 0)

            +

            (5 if month > 0 else 0)

            +

            (
                5 if breadth >= 60
                else 3 if breadth >= 45
                else 0
            )
        )

        sector_rows.append({

            "sector": sector,

            "w1": week,

            "m1": month,

            "breadth": breadth,

            "score": int(
                sector_score
            )
        })

    sector_df = pd.DataFrame(
        sector_rows
    ).sort_values(
        "score",
        ascending=False
    )

    sector_map = dict(
        zip(
            sector_df.sector,
            sector_df.score
        )
    )

    # =========================
    # FINAL STOCK RATING
    # =========================

    output = []

    for stock in stocks:

        sector_score = sector_map.get(
            stock["sector"],
            0
        )

        total = (
            sector_score +
            stock["trend"] +
            stock["momentum"] +
            stock["rs"] +
            stock["volscore"] +
            stock["setup"] +
            stock["risk"]
        )

        price = stock["close"]

        # =========================
        # ATR BASED STOP
        # =========================

        stop_distance = (
            stock["atr"] * 1.5
        )

        stop = price - stop_distance

        # Safety floor for extreme ATR
        stop_pct = (
            (price - stop) / price
            if price
            else 1
        )

        stop_pct = max(
            0.025,
            min(
                0.06,
                stop_pct
            )
        )

        stop = price * (
            1 - stop_pct
        )

        # =========================
        # 2.5R TARGET
        # =========================

        target = (
            price +
            (price - stop) * 2.5
        )

        rr = (
            (target - price) /
            (price - stop)
            if price > stop
            else 0
        )

        # =========================
        # STRICT BUY CONDITIONS
        # =========================

        strong_ema_trend = (
            stock["price_above_20"]
            and stock["ema20_above_50"]
            and stock["ema50_above_200"]
            and stock["ema200_rising"]
        )

        positive_momentum = (
            stock["w1"] > 0
            and stock["m1"] > 0
        )

        good_rsi = (
            55 <= stock["rsi"] <= 70
        )

        strong_volume = (
            stock["vol"] >= 1.5
        )

        valid_setup = (
            stock["breakout"]
            or stock["pullback"]
        )

        good_rr = (
            rr >= 2
        )

        # =========================
        # ACTION
        # =========================

       if (
    total >= 85
    and sector_score >= 10
    and strong_ema_trend
    and positive_momentum
    and valid_setup
    and good_rr
):

            action = "BUY NOW"

        elif (
            total >= 75
            and sector_score >= 10
            and strong_ema_trend
            and positive_momentum
            and good_rr
        ):

            action = "BUY / WATCH"

        elif total >= 60:

            action = "WAIT"

        else:

            action = "AVOID"

        setup_name = (
            "Breakout"
            if stock["breakout"]
            else "Pullback"
            if stock["pullback"]
            else "None"
        )

        output.append({

            "symbol": stock["symbol"],

            "sector": stock["sector"],

            "rating": int(
                min(100, total)
            ),

            "action": action,

            "entry": price,

            "stop": stop,

            "target": target,

            "rr": rr,

            "rsi": stock["rsi"],

            "vol": stock["vol"],

            "w1": stock["w1"],

            "m1": stock["m1"],

            "m3": stock["m3"],

            "ema20": stock["ema20"],

            "ema50": stock["ema50"],

            "ema200": stock["ema200"],

            "ema_trend": strong_ema_trend,

            "breakout": stock["breakout"],

            "pullback": stock["pullback"],

            "setup": setup_name
        })

    # =========================
    # SORT
    # =========================

    output.sort(
        key=lambda x: x["rating"],
        reverse=True
    )

    # =========================
    # SAVE DATA
    # =========================

    payload = {

        "updated":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),

        "stocks_scanned":
            len(output),

        "sectors":
            sector_df.to_dict(
                "records"
            ),

        "stocks":
            output
    }

    with open(
        "data.json",
        "w"
    ) as file:

        json.dump(
            payload,
            file,
            indent=2
        )

    print(
        "Scanned",
        len(output),
        "stocks; top rating",
        output[0]["rating"]
        if output
        else None
    )


if __name__ == "__main__":
    main()
