let DATA=null,filter="ALL";
const money=x=>x==null||!isFinite(x)?"—":"₹"+x.toFixed(2);
function actionClass(a){return a.startsWith("BUY")?"buy":a.startsWith("WATCH")?"watch":a==="WAIT"?"wait":"avoid"}
function render(){
 if(!DATA)return;
 document.querySelector("#updated").textContent="Updated: "+DATA.updated;
 document.querySelector("#sc").textContent=DATA.sectors.filter(x=>x.score>=12).length;
 document.querySelector("#nc").textContent=DATA.stocks_scanned;
 document.querySelector("#bc").textContent=DATA.stocks.filter(x=>x.action.startsWith("BUY")).length;
 document.querySelector("#tr").textContent=DATA.stocks.length?DATA.stocks[0].rating:"—";
 document.querySelector("#sectors").innerHTML=DATA.sectors.slice(0,15).map((x,i)=>`<tr><td>${i+1}</td><td><b>${x.sector}</b></td><td>${x.w1.toFixed(1)}%</td><td>${x.m1.toFixed(1)}%</td><td>${x.breadth.toFixed(0)}%</td><td><b>${x.score}</b></td></tr>`).join("");
 let rows=DATA.stocks.filter(x=>filter==="ALL"||(filter==="BUY"&&x.action.startsWith("BUY"))||(filter==="WATCH"&&x.action.includes("WATCH"))||(filter==="WAIT"&&x.action==="WAIT"));
 document.querySelector("#stocks").innerHTML=rows.slice(0,50).map(x=>`<tr><td><b>${x.symbol}</b></td><td>${x.sector}</td><td><b>${x.rating}</b></td><td class="${actionClass(x.action)}">${x.action}</td><td>${money(x.entry)}</td><td>${money(x.stop)}</td><td>${money(x.target)}</td><td>${x.rr.toFixed(1)}R</td><td>${x.rsi.toFixed(0)}</td><td>${x.vol.toFixed(1)}x</td><td>${x.w1.toFixed(1)}%</td><td>${x.m1.toFixed(1)}%</td><td>${x.setup}</td></tr>`).join("");
}
document.querySelectorAll(".filters button").forEach(b=>b.onclick=()=>{document.querySelectorAll(".filters button").forEach(z=>z.classList.remove("active"));b.classList.add("active");filter=b.dataset.f;render()});
fetch("data.json?"+Date.now()).then(r=>r.json()).then(d=>{DATA=d;render()}).catch(()=>document.querySelector("#updated").textContent="Data unavailable");