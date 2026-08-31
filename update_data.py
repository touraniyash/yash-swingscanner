import json, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests, yfinance as yf

CONSTITUENTS="https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
headers={"User-Agent":"Mozilla/5.0"}

def get_constituents():
    r=requests.get(CONSTITUENTS,headers=headers,timeout=30)
    r.raise_for_status()
    from io import BytesIO
    df=pd.read_csv(BytesIO(r.content))
    df.columns=[str(c).strip() for c in df.columns]
    # Current NSE file normally has Company Name, Industry, Symbol, Series, ISIN Code.
    symbol_col=next(c for c in df.columns if c.lower()=="symbol")
    industry_col=next((c for c in df.columns if c.lower()=="industry"),None)
    if industry_col is None: industry_col="Industry"
    out=df[[symbol_col,industry_col]].rename(columns={symbol_col:"symbol",industry_col:"sector"}).dropna()
    out["symbol"]=out["symbol"].astype(str).str.strip()
    out["sector"]=out["sector"].astype(str).str.strip()
    return out.drop_duplicates("symbol")

def rsi(series,n=14):
    d=series.diff()
    up=d.clip(lower=0).rolling(n).mean()
    dn=(-d.clip(upper=0)).rolling(n).mean()
    rs=up/dn.replace(0,np.nan)
    return (100-100/(1+rs)).fillna(50)

def score_stock(c,v):
    e20=c.ewm(span=20,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    e200=c.ewm(span=200,adjust=False).mean()
    rrsi=rsi(c)
    vr=(v/v.rolling(20).mean()).replace([np.inf,-np.inf],np.nan)
    w1=(c/c.shift(5)-1)*100
    m1=(c/c.shift(21)-1)*100
    m3=(c/c.shift(63)-1)*100
    atr=(c.diff().abs().rolling(14).mean()) # fallback volatility proxy
    close=float(c.iloc[-1]); e20v=float(e20.iloc[-1]); e50v=float(e50.iloc[-1]); e200v=float(e200.iloc[-1])
    rv=float(rrsi.iloc[-1]); vv=float(vr.iloc[-1]); w=float(w1.iloc[-1]); m=float(m1.iloc[-1]); m_3=float(m3.iloc[-1])
    trend=(5 if close>e20v else 0)+(5 if e20v>e50v else 0)+(5 if e50v>e200v else 0)+(5 if e200v>float(e200.iloc[-21]) else 0)+(5 if close>e50v else 0)
    momentum=(5 if w>0 else 0)+(5 if m>0 else 0)+(5 if m_3>0 else 0)
    rs=10 if 55<=rv<=68 else 7 if 50<=rv<55 or 68<rv<=72 else 3 if 45<=rv<50 else 0
    vol=10 if vv>=1.5 else 7 if vv>=1.2 else 4 if vv>=1 else 0
    prev20=float(c.shift(1).rolling(20).max().iloc[-1])
    breakout=close>prev20
    dist=abs(close-e20v)/e20v if e20v else 1
    pullback=close>e20v and dist<=.04 and float(c.iloc[-1])>=float(c.iloc[-2])
    setup=10 if breakout else 8 if pullback else 0
    atr_pct=float(c.pct_change().rolling(14).std().iloc[-1]*close) / close if close else 1
    risk=10 if atr_pct<=.025 else 7 if atr_pct<=.04 else 4 if atr_pct<=.06 else 0
    return dict(close=close,rsi=rv,vol=vv,w1=w,m1=m,m3=m_3,trend=trend,momentum=momentum,rs=rs,volscore=vol,setup=setup,risk=risk,breakout=breakout,pullback=pullback,atr_pct=atr_pct)

def main():
    u=get_constituents()
    symbols=u.symbol.tolist()
    tickers=[s+".NS" for s in symbols]
    stocks=[]
    # Batch downloads reduce the chance of Yahoo throttling a large request.
    for start in range(0,len(tickers),80):
        batch=tickers[start:start+80]
        try:
            raw=yf.download(batch,period="1y",interval="1d",auto_adjust=True,group_by="ticker",threads=True,progress=False)
        except Exception as e:
            print("batch error",e); continue
        for _,row in u.iloc[start:start+80].iterrows():
            s=row.symbol; sec=row.sector
            try:
                x=raw[s+".NS"].dropna()
                if len(x)<220: continue
                q=score_stock(x["Close"],x["Volume"])
                stocks.append({"symbol":s,"sector":sec,**q})
            except Exception as e:
                print("skip",s,e)
        time.sleep(1)

    df=pd.DataFrame(stocks)
    if df.empty: raise RuntimeError("No stock data downloaded")

    # Sector strength: 20 points = 5 trend + 5 1W + 5 1M + 5 breadth.
    secrows=[]
    for sec,g in df.groupby("sector"):
        w=float(g.w1.mean()); m=float(g.m1.mean())
        breadth=float((g.w1.gt(0)&g.m1.gt(0)).mean()*100)
        trendbreadth=float((g.trend>=15).mean()*100)
        s20=(5 if trendbreadth>=50 else 3 if trendbreadth>=35 else 0)+(5 if w>0 else 0)+(5 if m>0 else 0)+(5 if breadth>=60 else 3 if breadth>=45 else 0)
        secrows.append({"sector":sec,"w1":w,"m1":m,"breadth":breadth,"score":int(s20)})
    secdf=pd.DataFrame(secrows).sort_values("score",ascending=False)
    secmap=dict(zip(secdf.sector,secdf.score))

    out=[]
    for r in stocks:
        sector_score=secmap.get(r["sector"],0)
        # Exact 100: sector 20 + trend25 + momentum15 + RSI10 + volume10 + setup10 + risk10
        total=sector_score+r["trend"]+r["momentum"]+r["rs"]+r["volscore"]+r["setup"]+r["risk"]
        close=r["close"]
        # Practical swing levels. Entry is current close; stop is 1.5x a 14-day volatility proxy.
        risk_pct=max(0.025,min(0.06,r["atr_pct"]*1.5))
        stop=close*(1-risk_pct)
        target=close+(close-stop)*2.5
        rr=(target-close)/(close-stop) if close>stop else 0
        if total>=85 and sector_score>=12 and (r["breakout"] or r["pullback"]):
            action="BUY NOW"
        elif total>=75 and sector_score>=10:
            action="BUY / WATCH"
        elif total>=60:
            action="WAIT"
        else:
            action="AVOID"
        setup="Breakout" if r["breakout"] else "Pullback" if r["pullback"] else "None"
        out.append({"symbol":r["symbol"],"sector":r["sector"],"rating":int(min(100,total)),"action":action,"entry":close,"stop":stop,"target":target,"rr":rr,"rsi":r["rsi"],"vol":r["vol"],"w1":r["w1"],"m1":r["m1"],"setup":setup})
    out.sort(key=lambda z:z["rating"],reverse=True)
    payload={"updated":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),"stocks_scanned":len(out),"sectors":secdf.to_dict("records"),"stocks":out}
    with open("data.json","w") as f: json.dump(payload,f,indent=2)
    print("Scanned",len(out),"stocks; top rating",out[0]["rating"] if out else None)

if __name__=="__main__": main()
