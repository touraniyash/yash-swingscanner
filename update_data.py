import json
from datetime import datetime,timezone
import pandas as pd,numpy as np,yfinance as yf
u=pd.read_csv("symbols.csv"); tick=[s+".NS" for s in u.symbol]
raw=yf.download(tick,period="1y",interval="1d",auto_adjust=True,group_by="ticker",threads=True,progress=False)
stocks=[]; sec={}
for _,r in u.iterrows():
 s,sector=r.symbol,r.sector
 try:
  x=raw[s+".NS"].dropna(); c=x.Close; v=x.Volume
  if len(c)<220: continue
  e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean(); e200=c.ewm(span=200,adjust=False).mean()
  d=c.diff(); g=d.clip(lower=0).rolling(14).mean(); l=(-d.clip(upper=0)).rolling(14).mean(); rs=g/l.replace(0,np.nan); rsi=(100-100/(1+rs)).fillna(50)
  vr=(v/v.rolling(20).mean()).iloc[-1]; w=(c.iloc[-1]/c.iloc[-6]-1)*100; m=(c.iloc[-1]/c.iloc[-22]-1)*100; m3=(c.iloc[-1]/c.iloc[-64]-1)*100
  trend=c.iloc[-1]>e20.iloc[-1] and e20.iloc[-1]>e50.iloc[-1] and e50.iloc[-1]>e200.iloc[-1] and e200.iloc[-1]>e200.iloc[-21]
  br=c.iloc[-1]>c.shift(1).rolling(20).max().iloc[-1]; dist=abs(c.iloc[-1]-e20.iloc[-1])/e20.iloc[-1]; pb=c.iloc[-1]>e20.iloc[-1] and dist<=.05
  score=(30 if trend else 0)+(7 if 50<=rsi.iloc[-1]<=70 else 3 if rsi.iloc[-1]<=75 else 0)+(8 if vr>=1.2 else 4 if vr>=1 else 0)+(10 if w>0 and m>0 and m3>0 else 6 if w>0 and m>0 else 3 if w>0 else 0)+(5 if e20.iloc[-1]>e50.iloc[-1] else 0)+(8 if br else 0)+(7 if pb else 0)+(5 if dist<=.05 else 3 if dist<=.08 else 0)
  action="BUY NOW" if score>=90 and (br or pb) else "BUY / WATCH" if score>=80 else "WAIT" if score>=60 else "AVOID"
  stocks.append({"symbol":s,"sector":sector,"rating":int(min(100,score)),"action":action,"rsi":float(rsi.iloc[-1]),"vol":float(vr),"w1":float(w),"m1":float(m),"setup":"Breakout" if br else "Pullback" if pb else "None"})
  sec.setdefault(sector,[]).append((w,m))
 except Exception as e: print(s,e)
sectors=[]
for s,a in sec.items():
 w=np.mean([z[0] for z in a]); m=np.mean([z[1] for z in a]); b=np.mean([z[0]>0 and z[1]>0 for z in a])*100
 score=round(min(100,30+(10 if w>0 else 0)+(10 if m>0 else 0)+(20 if w>2 else 0)+(20 if m>5 else 0)+b*.2))
 sectors.append({"sector":s,"w1":float(w),"m1":float(m),"breadth":float(b),"score":score})
sectors.sort(key=lambda x:x["score"],reverse=True); stocks.sort(key=lambda x:x["rating"],reverse=True)
json.dump({"updated":datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),"sectors":sectors,"stocks":stocks},open("data.json","w"),indent=2)
