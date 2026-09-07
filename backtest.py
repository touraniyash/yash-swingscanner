"""
NSE Swing Strategy Backtest
Strategy:
- EMA 20 > EMA 50 > EMA 200 and close above EMA 20
- RSI between 50 and 68
- Volume >= 1.5x 20-day average
- Score >= 90 for the main test
- 2.5R target
- Compare 5, 10 and 15 trading-day maximum holding periods
- Also reports 90-100 and 80-89 score buckets

The script downloads daily NSE stock data from Yahoo Finance and writes:
    backtest_results.csv
"""

import math
import time
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()

TARGET_R = 2.5
HOLD_DAYS = [5, 10, 15]
MIN_SCORE = 80

# NIFTY 500 symbols. The list is intentionally kept in a plain Python list so
# the workflow does not depend on NSE's website being reachable.
# You can replace/extend this list later without changing the backtest engine.
NIFTY_SYMBOLS = [
    "360ONE","3MINDIA","ABB","ACC","AIAENG","APLAPOLLO","AUBANK","AWL",
    "AARTIIND","AAVAS","ABBOTINDIA","ABCAPITAL","ABFRL","ABREL","ABSLAMC",
    "ACE","ADANIENSOL","ADANIENT","ADANIGREEN","ADANIPORTS","ADANIPOWER",
    "ATGL","ABCAPITAL","ADANIWILMAR","ADFFOODS","AFFLE","AJANTAPHARM",
    "ALKEM","ALKYLAMINE","ALOKINDS","AMARAJABAT","AMBER","AMBUJACEM",
    "ANGELONE","ANANDRATHI","APARINDS","APLLTD","APOLLOHOSP","APOLLOTYRE",
    "APTUS","ARE&M","ASAHIINDIA","ASHOKLEY","ASIANPAINT","ASTERDM",
    "ASTRAL","ATUL","AUROPHARMA","AVANTIFEED","AXISBANK","BAJAJ-AUTO",
    "BAJAJFINSV","BAJFINANCE","BALKRISIND","BALRAMCHIN","BANDHANBNK",
    "BANKBARODA","BANKINDIA","BATAINDIA","BAYERCROP","BBTC","BDL","BEL",
    "BEML","BERGEPAINT","BHARATFORG","BHARTIARTL","BHEL","BIOCON",
    "BIRLACORPN","BLUEDART","BLUESTARCO","BOSCHLTD","BPCL","BRIGADE",
    "BRITANNIA","BSE","BSOFT","CAMPUS","CANBK","CANFINHOME","CAPPL",
    "CARBORUNIV","CASTROLIND","CEATLTD","CENTRALBK","CENTURYPLY",
    "CESC","CGPOWER","CHALET","CHAMBLFERT","CHEMPLASTS","CHENNPETRO",
    "CHOLAFIN","CIEINDIA","CIPLA","CLEAN","COALINDIA","COCHINSHIP",
    "COFORGE","COLPAL","CONCOR","COROMANDEL","CREDITACC","CROMPTON",
    "CUB","CUMMINSIND","CYIENT","DABUR","DALBHARAT","DATAPATTNS","DCMSHRIRAM",
    "DEEPAKNTR","DELHIVERY","DELTACORP","DEVYANI","DHANI","DIVISLAB",
    "DIXON","DLF","DMART","DRREDDY","EICHERMOT","EIDPARRY","EIHOTEL",
    "ELECON","ELGIEQUIP","EMAMILTD","ENDURANCE","ENGINERSIN","EPL",
    "EQUITASBNK","ERIS","ESCORTS","EXIDEIND","FACT","FEDERALBNK",
    "FINCABLES","FINEORG","FINPIPE","FIVESTAR","FLUOROCHEM","FORTIS",
    "FSL","GAIL","GESHIP","GICRE","GILLETTE","GLAND","GLAXO","GLENMARK",
    "GMRAIRPORT","GNFC","GODFRYPHLP","GODREJAGRO","GODREJCP","GODREJIND",
    "GODREJPROP","GRANULES","GRAPHITE","GRASIM","GRINDWELL","GUJALKALI",
    "GUJGASLTD","HAL","HAPPSTMNDS","HAVELLS","HCLTECH","HDFCAMC","HDFCBANK",
    "HDFCLIFE","HEG","HEROMOTOCO","HFCL","HINDALCO","HINDCOPPER","HINDPETRO",
    "HINDUNILVR","HINDZINC","HOMEFIRST","HONASA","HUDCO","ICICIBANK",
    "ICICIGI","ICICIPRULI","IDBI","IDEA","IDFCFIRSTB","IEX","IFBIND",
    "IGL","IIFL","INDHOTEL","INDIACEM","INDIAMART","INDIANB","INDIGO",
    "INDUSINDBK","INDUSTOWER","INFY","INOXWIND","INTELLECT","IOB","IOC",
    "IPCALAB","IRB","IRCON","IRCTC","IREDA","IRFC","ITC","ITI","JINDALSAW",
    "JINDALSTEL","JKCEMENT","JKLAKSHMI","JKPAPER","JMFINANCIL","JSL",
    "JSWENERGY","JSWSTEEL","JUBLFOOD","JUBLPHARMA","JUSTDIAL","JYOTHYLAB",
    "KAJARIACER","KALYANKJIL","KANSAINER","KARURVYSYA","KEC","KEI",
    "KFINTECH","KIRLOSBROS","KIRLOSENG","KNRCON","KPIL","KPITTECH",
    "KRBL","LALPATHLAB","LATENTVIEW","LAURUSLABS","LEMONTREE","LICHSGFIN",
    "LICI","LINDEINDIA","LODHA","LT","LTIM","LTTS","LUPIN","M&M","M&MFIN",
    "MAHABANK","MAHINDCIE","MAHLOG","MANAPPURAM","MANKIND","MARICO",
    "MARUTI","MASTEK","MAXHEALTH","MAZDOCK","MCX","MEDANTA","METROBRAND",
    "METROPOLIS","MFSL","MGL","MIDHANI","MINDACORP","MINDSPACE","MMTC",
    "MOIL","MOTHERSON","MOTILALOFS","MPHASIS","MRF","MRPL","MSUMI",
    "MUTHOOTFIN","NATIONALUM","NAUKRI","NAVINFLUOR","NBCC","NCC","NESTLEIND",
    "NETWORK18","NHPC","NIACL","NLCINDIA","NMDC","NOCIL","NTPC","NUVOCO",
    "OBEROIRLTY","OFSS","OIL","OLECTRA","ONGC","PAGEIND","PATANJALI",
    "PAYTM","PCBL","PERSISTENT","PETRONET","PFC","PFIZER","PGHH","PHOENIXLTD",
    "PIDILITIND","PIIND","PNB","PNBHOUSING","POLYCAB","POLYMED","POWERGRID",
    "POWERMECH","PPLPHARMA","PRAJIND","PRESTIGE","PRICOLLTD","PRINCEPIPE",
    "PRIVISCL","PVRINOX","QUESS","RADICO","RAILTEL","RAIN","RAJESHEXPO",
    "RALLIS","RAMCOCEM","RBLBANK","RECLTD","REDINGTON","RELAXO","RELIANCE",
    "RENUKA","RHIM","RITES","RKFORGE","ROUTE","RPOWER","SAGCEM","SAIL",
    "SAMMAANCAP","SAPPHIRE","SBICARD","SBILIFE","SBIN","SCHAEFFLER",
    "SCHNEIDER","SCI","SHARDACROP","SHAREINDIA","SHREECEM","SHRIRAMFIN",
    "SHYAMMETL","SIEMENS","SJVN","SKFINDIA","SOBHA","SOLARINDS","SONACOMS",
    "SONATSOFTW","SRF","STARHEALTH","STARCEMENT","SUMICHEM","SUNPHARMA",
    "SUNTV","SUPREMEIND","SURYAROSNI","SUZLON","SWANENERGY","SWSOLAR",
    "TATACHEM","TATACOMM","TATACONSUM","TATAELXSI","TATAMOTORS","TATAPOWER",
    "TATASTEEL","TATATECH","TCI","TCS","TECHM","TECHNOE","TEJASNET",
    "THERMAX","TIINDIA","TIMKEN","TITAN","TORNTPHARM","TORNTPOWER",
    "TRENT","TRIDENT","TRITURBINE","TVSMOTOR","UBL","UCOBANK","UJJIVANSFB",
    "ULTRACEMCO","UNIONBANK","UNITDSPR","UNOMINDA","UPL","USHAMART","UTIAMC",
    "VAIBHAVGBL","VARROC","VEDL","VGUARD","VIJAYA","VINATIORGA","VIPIND",
    "VOLTAS","VSTIND","WELCORP","WELSPUNLIV","WESTLIFE","WHIRLPOOL",
    "WIPRO","YESBANK","ZEEL","ZENSARTECH","ZFCVINDIA","ZOMATO","ZYDUSLIFE"
]

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def score_row(row):
    score = 0
    # Trend / EMA structure: 50 points
    if row["Close"] > row["EMA20"]:
        score += 15
    if row["EMA20"] > row["EMA50"]:
        score += 15
    if row["EMA50"] > row["EMA200"]:
        score += 20

    # RSI: 20 points
    if 50 <= row["RSI"] <= 68:
        score += 20
    elif 45 <= row["RSI"] <= 72:
        score += 10

    # Volume: 20 points
    if row["VolumeRatio"] >= 1.5:
        score += 20
    elif row["VolumeRatio"] >= 1.2:
        score += 10

    # Momentum: 10 points
    if row["Close"] > row["Close_5D"]:
        score += 5
    if row["Close"] > row["Close_20D"]:
        score += 5
    return score

def prepare(df):
    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in needed):
        return None

    df = df[needed].copy().dropna()
    if len(df) < 220:
        return None

    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["RSI"] = rsi(df["Close"], 14)
    df["VolumeAvg20"] = df["Volume"].rolling(20).mean()
    df["VolumeRatio"] = df["Volume"] / df["VolumeAvg20"]
    df["Close_5D"] = df["Close"].shift(5)
    df["Close_20D"] = df["Close"].shift(20)
    df["Score"] = df.apply(score_row, axis=1)

    df["Signal"] = (
        (df["Score"] >= MIN_SCORE)
        & (df["Close"] > df["EMA20"])
        & (df["EMA20"] > df["EMA50"])
        & (df["EMA50"] > df["EMA200"])
        & (df["RSI"].between(50, 68))
        & (df["VolumeRatio"] >= 1.5)
    )
    return df.dropna(subset=["EMA200", "RSI", "VolumeRatio"])

def simulate_trade(df, signal_i, hold_days):
    entry_i = signal_i + 1
    if entry_i >= len(df):
        return None

    entry = float(df.iloc[entry_i]["Open"])
    signal_low = float(df.iloc[signal_i]["Low"])

    # Use the signal candle low as the initial stop.
    risk = entry - signal_low
    if not np.isfinite(risk) or risk <= 0:
        return None

    stop = signal_low
    target = entry + TARGET_R * risk
    last_i = min(entry_i + hold_days - 1, len(df) - 1)

    exit_i = last_i
    exit_price = float(df.iloc[last_i]["Close"])
    exit_reason = "TIME"

    for j in range(entry_i, last_i + 1):
        day = df.iloc[j]
        day_low = float(day["Low"])
        day_high = float(day["High"])

        # Conservative assumption when both are touched intraday:
        # stop is considered hit first.
        if day_low <= stop:
            exit_i = j
            exit_price = stop
            exit_reason = "STOP"
            break

        if day_high >= target:
            exit_i = j
            exit_price = target
            exit_reason = "TARGET"
            break

    r_multiple = (exit_price - entry) / risk
    return {
        "EntryDate": df.index[entry_i].date().isoformat(),
        "ExitDate": df.index[exit_i].date().isoformat(),
        "Entry": round(entry, 2),
        "Stop": round(stop, 2),
        "Target": round(target, 2),
        "Exit": round(exit_price, 2),
        "R": round(r_multiple, 4),
        "ReturnPct": round((exit_price / entry - 1) * 100, 4),
        "HoldDays": int(exit_i - entry_i + 1),
        "ExitReason": exit_reason,
    }

def backtest_symbol(symbol):
    ticker = symbol + ".NS"
    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        df = prepare(df)
        if df is None:
            return []

        trades = []
        last_signal_i = -999

        for i in range(len(df) - 1):
            if not bool(df.iloc[i]["Signal"]):
                continue

            # Avoid stacking a new position while the previous signal's
            # 15-day test window is still active.
            if i - last_signal_i < 15:
                continue

            for hold in HOLD_DAYS:
                trade = simulate_trade(df, i, hold)
                if trade is None:
                    continue

                trade.update({
                    "Symbol": symbol,
                    "SignalDate": df.index[i].date().isoformat(),
                    "Score": int(df.iloc[i]["Score"]),
                    "RSI": round(float(df.iloc[i]["RSI"]), 2),
                    "VolumeRatio": round(float(df.iloc[i]["VolumeRatio"]), 2),
                    "HoldPlan": hold,
                })
                trades.append(trade)

            last_signal_i = i

        return trades

    except Exception as exc:
        print(f"{symbol}: skipped ({exc})")
        return []

def summarize(trades_df):
    if trades_df.empty:
        return pd.DataFrame()

    rows = []
    for (hold, bucket), g in trades_df.groupby(["HoldPlan", "ScoreBucket"]):
        r = g["R"]
        wins = (r > 0).sum()
        losses = (r <= 0).sum()
        gross_profit = r[r > 0].sum()
        gross_loss = abs(r[r < 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else math.inf
        rows.append({
            "HoldPlan": hold,
            "ScoreBucket": bucket,
            "Trades": len(g),
            "Wins": int(wins),
            "Losses": int(losses),
            "WinRatePct": round(wins / len(g) * 100, 2),
            "TotalR": round(r.sum(), 2),
            "AverageR": round(r.mean(), 3),
            "AverageReturnPct": round(g["ReturnPct"].mean(), 3),
            "ProfitFactor": round(pf, 3) if math.isfinite(pf) else "inf",
            "MaxDrawdownR": round(max_drawdown(g["R"]), 2),
        })
    return pd.DataFrame(rows)

def max_drawdown(r_series):
    equity = r_series.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min())

def main():
    print("=" * 60)
    print("NSE SWING STRATEGY BACKTEST")
    print("=" * 60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Target: {TARGET_R}R")
    print("Hold plans:", HOLD_DAYS)
    print("Score buckets: 90-100 and 80-89")
    print()

    all_trades = []
    total = len(NIFTY_SYMBOLS)

    for n, symbol in enumerate(dict.fromkeys(NIFTY_SYMBOLS), 1):
        print(f"[{n}/{total}] {symbol}")
        trades = backtest_symbol(symbol)
        all_trades.extend(trades)
        time.sleep(0.05)

    results = pd.DataFrame(all_trades)

    if results.empty:
        # Always create the artifact so GitHub Actions never fails because
        # the CSV is missing.
        results = pd.DataFrame(columns=[
            "Symbol", "SignalDate", "EntryDate", "ExitDate", "Score",
            "ScoreBucket", "RSI", "VolumeRatio", "HoldPlan", "Entry",
            "Stop", "Target", "Exit", "R", "ReturnPct", "HoldDays",
            "ExitReason"
        ])
        summary = pd.DataFrame()
    else:
        results["ScoreBucket"] = np.where(
            results["Score"] >= 90, "90-100",
            np.where(results["Score"] >= 80, "80-89", "Below 80")
        )
        results = results[results["Score"] >= MIN_SCORE].copy()
        results = results.sort_values(
            ["HoldPlan", "SignalDate", "Score", "Symbol"]
        )
        summary = summarize(results)

    results.to_csv("backtest_results.csv", index=False)
    summary.to_csv("backtest_summary.csv", index=False)

    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    print(f"Total trade rows: {len(results)}")
    if not summary.empty:
        print(summary.to_string(index=False))
    print()
    print("Results saved to backtest_results.csv")
    print("Summary saved to backtest_summary.csv")

if __name__ == "__main__":
    main()
