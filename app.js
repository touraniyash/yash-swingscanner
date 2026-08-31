let DATA=null,filter="ALL";
const money=x=>x==null||!isFinite(x)?"—":"₹"+Number(x).toFixed(2);
const cls=a=>a.startsWith("BUY NOW")?"buy":a.includes("WATCH")?"watch":a==="WAIT"?"wait":"avoid";
function render(){
 if(!DATA)return;
 document.querySelector("#updated").textContent="Updated: "+DATA.updated;
 document.querySelector("#sc").textContent=DATA.sectors.filter(x=>x.score>=12).length;
 document.querySelector("#nc").textContent=DATA.stocks_scanned;
 document.querySelector("#bc").textContent=DATA.stocks.filter(x=>x.action==="BUY NOW").length;
 document.querySelector("#tr").textContent=DATA.stocks.length?DATA.stocks[0].rating:"—";
 document.querySelector("#sectors").innerHTML=DATA.sectors.slice(0,15).map((x,i)=>`<tr><td>${i+1}</td><td><b>${x.sector}</b></td><td>${Number(x.w1).toFixed(1)}%</td><td>${Number(x.m1).toFixed(1)}%</td><td>${Number(x.breadth).toFixed(0)}%</td><td><b>${x.score}/20</b></td></tr>`).join("");
 const top=DATA.stocks.slice(0,10);
 document.querySelector("#topcards").innerHTML=top.length?top.map(x=>`<div class="setupcard"><div class="setuphead"><div><b>${x.symbol}</b> · ${x.sector}<br><span class="${cls(x.action)}">${x.action}</span> · ${x.setup}</div><div class="rating">${x.rating}/100</div></div><div class="metrics"><div class="metric"><small>Entry</small><b>${money(x.entry)}</b></div><div class="metric"><small>Stop</small><b>${money(x.stop)}</b></div><div class="metric"><small>Target</small><b>${money(x.target)}</b></div><div class="metric"><small>R:R</small><b>${Number(x.rr).toFixed(1)}R</b></div><div class="metric"><small>RSI</small><b>${Number(x.rsi).toFixed(0)}</b></div><div class="metric"><small>Volume</small><b>${Number(x.vol).toFixed(1)}x</b></div><div class="metric"><small>1W / 1M</small><b>${Number(x.w1).toFixed(1)}% / ${Number(x.m1).toFixed(1)}%</b></div></div></div>`).join(""):`<div class="empty">No screened stocks yet. Run the Update swing scanner workflow.</div>`;
 let rows=DATA.stocks.filter(x=>filter==="ALL"||(filter==="BUY NOW"&&x.action==="BUY NOW")||(filter==="WATCH"&&x.action.includes("WATCH"))||(filter==="WAIT"&&x.action==="WAIT"));
 document.querySelector("#stocks").innerHTML=rows.slice(0,100).map(x=>`<tr><td><b>${x.symbol}</b></td><td>${x.sector}</td><td><b>${x.rating}</b></td><td class="${cls(x.action)}">${x.action}</td><td>${money(x.entry)}</td><td>${money(x.stop)}</td><td>${money(x.target)}</td><td>${Number(x.rr).toFixed(1)}R</td><td>${Number(x.rsi).toFixed(0)}</td><td>${Number(x.vol).toFixed(1)}x</td><td>${Number(x.w1).toFixed(1)}%</td><td>${Number(x.m1).toFixed(1)}%</td><td>${x.setup}</td></tr>`).join("");
}
document.querySelectorAll(".filters button").forEach(b=>b.onclick=()=>{document.querySelectorAll(".filters button").forEach(z=>z.classList.remove("active"));b.classList.add("active");filter=b.dataset.f;render()});
fetch("data.json?"+Date.now()).then(r=>r.json()).then(d=>{DATA=d;render()}).catch(()=>document.querySelector("#updated").textContent="Data unavailable");
