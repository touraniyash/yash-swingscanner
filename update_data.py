import json
import time
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf


CONSTITUENTS_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.niftyindices.com/",
}

BATCH_SIZE = 40
MIN_HISTORY = 220


def get_constituents():
    """Download the current NIFTY 500 constituent list."""
    response = requests.get(
        CONSTITUENTS_URL,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    df = pd.read_csv(BytesIO(response.content))
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = next(
        (c for c in df.columns if c.lower() == "symbol"),
        None,
    )

    industry_col = next(
        (c for c in df.columns if c.lower() == "industry"),
        None,
    )

    # Some versions of the constituent file may use a different
    # industry/sector column name.
    if industry_col is None:
        industry_col = next(
            (
                c
                for c in df.columns
                if c.lower() in {"sector", "industry name"}
            ),
            None,
        )

    if symbol_col is None:
        raise RuntimeError(
            f"Could not find Symbol column. Columns received: {list(df.columns)}"
        )

    if industry_col is None:
        # Keep the scanner working even if the source changes its
        # industry column name. Sector will simply be Unknown.
        df["Industry"] = "Unknown"
        industry_col = "Industry"

    out = df[[symbol_col, industry_col]].rename(
        columns={
            symbol_col: "symbol",
            industry_col: "sector",
        }
    )

    out = out.dropna()
    out["symbol"] = out["symbol"].astype(str).str.strip().str.upper()
    out["sector"] = out["sector"].astype(str).str.strip()

    out = out[
        (out["symbol"] != "")
        & (out["symbol"].str.lower() != "nan")
    ]

    return out.drop_duplicates("symbol").reset_index(drop=True)


def rsi(series, n=14):
    """Wilder-style RSI using rolling averages."""
    delta = series.diff()

    gain = delta.clip(lower=0).rolling(n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).rolling(n, min_periods=n).mean()

    rs = gain / loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    # If there has been no loss, RSI is 100.
    result = result.mask((loss == 0) & (gain > 0), 100)

    # If there has been no gain, RSI is 0.
    result = result.mask((gain == 0) & (loss > 0), 0)

    return result.fillna(50)


def calculate_atr(high, low, close, n=14):
    """Average True Range."""
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(n, min_periods=n).mean()


def clean_ohlcv(data):
    """Clean a ticker's OHLCV dataframe."""
    if data is None or data.empty:
        return pd.DataFrame()

    data = data.copy()

    # yfinance can occasionally return columns with an extra level.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            str(col[-1] if isinstance(col, tuple) else col)
            for col in data.columns
        ]

    required = ["High", "Low", "Close", "Volume"]

    if not all(col in data.columns for col in required):
        return pd.DataFrame()

    data = data[required].copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    return data


def extract_ticker_data(raw, ticker):
    """
    Extract one ticker from yfinance output.

    Handles both:
    - MultiIndex output from multi-ticker downloads
    - Normal columns from single-ticker downloads
    """
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Multi-ticker yfinance output.
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).astype(str)
        level1 = raw.columns.get_level_values(1).astype(str)

        if ticker in level0:
            data = raw[ticker].copy()
            return clean_ohlcv(data)

        if ticker in level1:
            data = raw.xs(ticker, axis=1, level=1).copy()
            return clean_ohlcv(data)

        return pd.DataFrame()

    # Single-ticker output.
    return clean_ohlcv(raw)


def download_batch(tickers):
    """
    Download a batch. If the batch fails, retry ticker-by-ticker.

    This makes the scanner much more resistant to Yahoo Finance
    rate limits and occasional batch failures.
    """
    if not tickers:
        return {}

    result = {}

    try:
        raw = yf.download(
            tickers,
            period="2y",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=False,
            progress=False,
            timeout=30,
        )

        if raw is not None and not raw.empty:
            for ticker in tickers:
                data = extract_ticker_data(raw, ticker)
                if not data.empty:
                    result[ticker] = data

    except Exception as exc:
        print("Batch download error:", exc)

    # Retry only tickers that were not returned by the batch.
    missing = [ticker for ticker in tickers if ticker not in result]

    if missing:
        print(f"Retrying {len(missing)} tickers individually...")

    for ticker in missing:
        try:
            raw = yf.download(
                ticker,
                period="2y",
                interval="1d",
                auto_adjust=True,
                group_by="column",
                threads=False,
                progress=False,
                timeout=30,
            )

            data = extract_ticker_data(raw, ticker)

            if not data.empty:
                result[ticker] = data

        except Exception as exc:
            print("Individual download failed:", ticker, exc)

        time.sleep(0.25)

    return result


def score_stock(high, low, close, volume):
    """Calculate the stock's technical score out of 80 points."""
    # =========================
    # EMA TREND
    # =========================
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()

    # =========================
    # RSI
    # =========================
    rrsi = rsi(close)

    # =========================
    # VOLUME
    # =========================
    volume_average = volume.rolling(20, min_periods=20).mean()

    volume_ratio = (
        volume / volume_average
    ).replace([np.inf, -np.inf], np.nan)

    # =========================
    # MOMENTUM
    # =========================
    w1 = (close / close.shift(5) - 1) * 100
    m1 = (close / close.shift(21) - 1) * 100
    m3 = (close / close.shift(63) - 1) * 100

    # =========================
    # ATR
    # =========================
    atr = calculate_atr(high, low, close)

    # Make sure the values needed below actually exist.
    if len(close) < MIN_HISTORY:
        raise ValueError(f"Only {len(close)} rows of history")

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

    values = [
        price,
        e20,
        e50,
        e200,
        rsi_value,
        volume_value,
        week,
        month,
        three_month,
        atr_value,
    ]

    if not all(np.isfinite(v) for v in values):
        raise ValueError("Insufficient valid technical indicator data")

    # =========================
    # EMA CONDITIONS
    # =========================
    price_above_20 = price > e20
    ema20_above_50 = e20 > e50
    ema50_above_200 = e50 > e200

    ema200_rising = (
        e200 > float(ema200.iloc[-21])
    )

    price_above_50 = price > e50

    # =========================
    # TREND SCORE /25
    # =========================
    trend = (
        (5 if price_above_20 else 0)
        + (5 if ema20_above_50 else 0)
        + (5 if ema50_above_200 else 0)
        + (5 if ema200_rising else 0)
        + (5 if price_above_50 else 0)
    )

    # =========================
    # MOMENTUM SCORE /15
    # =========================
    momentum = (
        (5 if week > 0 else 0)
        + (5 if month > 0 else 0)
        + (5 if three_month > 0 else 0)
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
        close.shift(1).rolling(20).max().iloc[-1]
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
        and price >= float(close.iloc[-2])
    )

    setup_score = (
        10
        if breakout
        else 8
        if pullback
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
        "pullback": pullback,
    }


def main():
    print("Starting swing scanner...")

    constituents = get_constituents()

    if constituents.empty:
        raise RuntimeError("NIFTY 500 constituent list is empty")

    symbols = constituents["symbol"].tolist()

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    print(f"Found {len(tickers)} NIFTY 500 symbols")

    stocks = []

    # =========================
    # DOWNLOAD DATA
    # =========================
    for start in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[start:start + BATCH_SIZE]

        print(
            f"Downloading batch "
            f"{start + 1}-{min(start + BATCH_SIZE, len(tickers))} "
            f"of {len(tickers)}..."
        )

        downloaded = download_batch(batch)

        if not downloaded:
            print("No data returned for this batch")
            continue

        batch_constituents = constituents.iloc[
            start:start + BATCH_SIZE
        ]

        for _, row in batch_constituents.iterrows():
            symbol = str(row["symbol"])
            sector = str(row["sector"])
            ticker = symbol + ".NS"

            data = downloaded.get(ticker)

            if data is None or data.empty:
                print("Skip", symbol, "- no usable data")
                continue

            if len(data) < MIN_HISTORY:
                print(
                    "Skip",
                    symbol,
                    f"- only {len(data)} rows"
                )
                continue

            try:
                result = score_stock(
                    data["High"],
                    data["Low"],
                    data["Close"],
                    data["Volume"],
                )

                stocks.append(
                    {
                        "symbol": symbol,
                        "sector": sector,
                        **result,
                    }
                )

            except Exception as exc:
                print("Skip", symbol, exc)

        # Small pause to reduce Yahoo Finance throttling.
        time.sleep(1)

    print(f"Technical scan completed: {len(stocks)} stocks")

    if not stocks:
        raise RuntimeError(
            "No stock data downloaded. "
            "Yahoo Finance may be temporarily unavailable or rate-limited."
        )

    df = pd.DataFrame(stocks)

    # =========================
    # SECTOR STRENGTH
    # =========================
    sector_rows = []

    for sector, group in df.groupby("sector"):
        week = float(group["w1"].mean())
        month = float(group["m1"].mean())

        breadth = float(
            (
                group["w1"].gt(0)
                & group["m1"].gt(0)
            ).mean() * 100
        )

        trend_breadth = float(
            (group["trend"] >= 15).mean() * 100
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
            (5 if breadth >= 60
             else 3 if breadth >= 45
             else 0)
        )

        sector_rows.append(
            {
                "sector": sector,
                "w1": week,
                "m1": month,
                "breadth": breadth,
                "score": int(sector_score),
            }
        )

    sector_df = pd.DataFrame(sector_rows)

    if sector_df.empty:
        raise RuntimeError("Unable to calculate sector strength")

    sector_df = sector_df.sort_values(
        "score",
        ascending=False,
    )

    sector_map = dict(
        zip(
            sector_df["sector"],
            sector_df["score"],
        )
    )

    # =========================
    # FINAL STOCK RATING
    # =========================
    output = []

    for stock in stocks:
        sector_score = int(
            sector_map.get(stock["sector"], 0)
        )

        total = (
            sector_score
            + stock["trend"]
            + stock["momentum"]
            + stock["rs"]
            + stock["volscore"]
            + stock["setup"]
            + stock["risk"]
        )

        price = float(stock["close"])

        # =========================
        # ATR BASED STOP
        # =========================
        stop_distance = stock["atr"] * 1.5
        stop = price - stop_distance

        # Safety floor for extreme ATR:
        # stop distance stays between 2.5% and 6%.
        stop_pct = (
            (price - stop) / price
            if price
            else 1
        )

        stop_pct = max(
            0.025,
            min(0.06, stop_pct),
        )

        stop = price * (1 - stop_pct)

        # =========================
        # 2.5R TARGET
        # =========================
        target = (
            price
            + (price - stop) * 2.5
        )

        rr = (
            (target - price) / (price - stop)
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
            and sector_score >= 12
            and strong_ema_trend
            and positive_momentum
            and good_rsi
            and strong_volume
            and valid_setup
            and good_rr
        ):
            action = "BUY NOW"

        elif (
            total >= 75
            and sector_score >= 10
            and strong_ema_trend
            and positive_momentum
            and good_rsi
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

        output.append(
            {
                "symbol": stock["symbol"],
                "sector": stock["sector"],
                "rating": int(min(100, total)),
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
                "setup": setup_name,
            }
        )

    # =========================
    # SORT
    # =========================
    output.sort(
        key=lambda x: x["rating"],
        reverse=True,
    )

    # =========================
    # SAVE DATA
    # =========================
    payload = {
        "updated": datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC"),
        "stocks_scanned": len(output),
        "sectors": sector_df.to_dict(
            orient="records"
        ),
        "stocks": output,
    }

    with open(
        "data.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            allow_nan=False,
        )

    print(
        "Scanned",
        len(output),
        "stocks; top rating",
        output[0]["rating"] if output else None,
    )


if __name__ == "__main__":
    main()
