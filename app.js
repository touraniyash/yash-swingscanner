let DATA = null, filter = "ALL";

const money = x =>
  x == null || !isFinite(x) ? "—" : "₹" + Number(x).toFixed(2);

const num = (x, digits = 2) =>
  x == null || !isFinite(x) ? "—" : Number(x).toFixed(digits);

const cls = a =>
  String(a || "").startsWith("BUY NOW")
    ? "buy"
    : String(a || "").includes("WATCH")
    ? "watch"
    : a === "WAIT"
    ? "wait"
    : "avoid";

// V4 CAPITAL & RISK SETTINGS
let capital = 100000;
let riskPercent = 1;
let maxAllocationPercent = 20;

function positionSize(x) {
  const entry = Number(x.entry);
  const stop = Number(x.stop);

  if (!isFinite(entry) || !isFinite(stop) || entry <= 0 || stop >= entry) {
    return {
      qty: 0,
      capitalUsed: 0,
      risk: 0,
      profit: 0
    };
  }

  const riskAmount = capital * (riskPercent / 100);
  const riskPerShare = entry - stop;

  const qtyByRisk = Math.floor(riskAmount / riskPerShare);

  const maxAllocation = capital * (maxAllocationPercent / 100);
  const qtyByCapital = Math.floor(maxAllocation / entry);

  const qty = Math.max(0, Math.min(qtyByRisk, qtyByCapital));

  const capitalUsed = qty * entry;
  const risk = qty * riskPerShare;

  const target = Number(x.target);
  const profit =
    isFinite(target) ? qty * Math.max(0, target - entry) : 0;

  return {
    qty,
    capitalUsed,
    risk,
    profit
  };
}

function injectV4Controls() {
  const topcards = document.querySelector("#topcards");
  if (!topcards || document.querySelector("#v4controls")) return;

  const box = document.createElement("div");
  box.id = "v4controls";
  box.style.cssText = `
    margin:16px 0;
    padding:16px;
    border:1px solid #ddd;
    border-radius:12px;
    background:#fafafa;
  `;

  box.innerHTML = `
    <div style="font-size:18px;font-weight:700;margin-bottom:10px;">
      V4 Trade Risk & Position Sizing
    </div>

    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:end;">
      <label>
        <small>Capital</small><br>
        <input id="v4capital" type="number" value="${capital}"
          style="padding:8px;width:140px;">
      </label>

      <label>
        <small>Risk / Trade %</small><br>
        <input id="v4risk" type="number" value="${riskPercent}"
          step="0.1" min="0.1"
          style="padding:8px;width:120px;">
      </label>

      <label>
        <small>Max Allocation %</small><br>
        <input id="v4allocation" type="number"
          value="${maxAllocationPercent}"
          step="1" min="1"
          style="padding:8px;width:140px;">
      </label>

      <button id="v4apply"
        style="padding:9px 16px;cursor:pointer;">
        Apply
      </button>
    </div>

    <div id="v4summary" style="margin-top:12px;font-size:14px;"></div>
  `;

  topcards.parentNode.insertBefore(box, topcards);

  document.querySelector("#v4apply").onclick = () => {
    capital = Number(document.querySelector("#v4capital").value) || 100000;
    riskPercent = Number(document.querySelector("#v4risk").value) || 1;
    maxAllocationPercent =
      Number(document.querySelector("#v4allocation").value) || 20;

    render();
  };
}

function render() {
  if (!DATA) return;

  injectV4Controls();

  document.querySelector("#updated").textContent =
    "Updated: " + DATA.updated;

  document.querySelector("#sc").textContent =
    DATA.sectors.filter(x => x.score >= 12).length;

  document.querySelector("#nc").textContent =
    DATA.stocks_scanned;

  document.querySelector("#bc").textContent =
    DATA.stocks.filter(x => x.action === "BUY NOW").length;

  document.querySelector("#tr").textContent =
    DATA.stocks.length ? DATA.stocks[0].rating : "—";

  // STRONGEST SECTORS
  document.querySelector("#sectors").innerHTML =
    DATA.sectors.slice(0, 15).map((x, i) => `
      <tr>
        <td>${i + 1}</td>
        <td><b>${x.sector}</b></td>
        <td>${num(x.w1, 1)}%</td>
        <td>${num(x.m1, 1)}%</td>
        <td>${num(x.breadth, 0)}%</td>
        <td><b>${x.score}/20</b></td>
      </tr>
    `).join("");

  // TOP SWING SETUPS
  const top = DATA.stocks.slice(0, 10);

  document.querySelector("#topcards").innerHTML =
    top.length
      ? top.map(x => {

          const p = positionSize(x);

          return `
            <div class="setupcard">

              <div class="setuphead">
                <div>
                  <b>${x.symbol}</b> · ${x.sector}<br>
                  <span class="${cls(x.action)}">${x.action}</span>
                  · ${x.setup}
                </div>

                <div class="rating">
                  ${x.rating}/100
                </div>
              </div>

              <div class="metrics">

                <div class="metric">
                  <small>Entry</small>
                  <b>${money(x.entry)}</b>
                </div>

                <div class="metric">
                  <small>Stop</small>
                  <b>${money(x.stop)}</b>
                </div>

                <div class="metric">
                  <small>Target</small>
                  <b>${money(x.target)}</b>
                </div>

                <div class="metric">
                  <small>R:R</small>
                  <b>${num(x.rr, 1)}R</b>
                </div>

                <div class="metric">
                  <small>RSI</small>
                  <b>${num(x.rsi, 0)}</b>
                </div>

                <div class="metric">
                  <small>Volume</small>
                  <b>${num(x.vol, 1)}x</b>
                </div>

                <div class="metric">
                  <small>1W / 1M</small>
                  <b>${num(x.w1, 1)}% / ${num(x.m1, 1)}%</b>
                </div>

                <div class="metric">
                  <small>Qty</small>
                  <b>${p.qty || "—"}</b>
                </div>

                <div class="metric">
                  <small>Capital Used</small>
                  <b>${p.qty ? money(p.capitalUsed) : "—"}</b>
                </div>

                <div class="metric">
                  <small>Risk</small>
                  <b>${p.qty ? money(p.risk) : "—"}</b>
                </div>

                <div class="metric">
                  <small>Target Profit</small>
                  <b>${p.qty ? money(p.profit) : "—"}</b>
                </div>

              </div>
            </div>
          `;
        }).join("")
      : `
        <div class="empty">
          No screened stocks yet.
          Run the Update swing scanner workflow.
        </div>
      `;

  // STOCK TABLE
  const rows = DATA.stocks.filter(x =>
    filter === "ALL" ||
    (filter === "BUY NOW" && x.action === "BUY NOW") ||
    (filter === "WATCH" && x.action.includes("WATCH")) ||
    (filter === "WAIT" && x.action === "WAIT")
  );

  document.querySelector("#stocks").innerHTML =
    rows.slice(0, 100).map(x => `
      <tr>
        <td><b>${x.symbol}</b></td>
        <td>${x.sector}</td>
        <td><b>${x.rating}</b></td>
        <td class="${cls(x.action)}">${x.action}</td>
        <td>${money(x.entry)}</td>
        <td>${money(x.stop)}</td>
        <td>${money(x.target)}</td>
        <td>${num(x.rr, 1)}R</td>
        <td>${num(x.rsi, 0)}</td>
        <td>${num(x.vol, 1)}x</td>
        <td>${num(x.w1, 1)}%</td>
        <td>${num(x.m1, 1)}%</td>
        <td>${x.setup}</td>
      </tr>
    `).join("");

  const summary = document.querySelector("#v4summary");

  if (summary) {
    summary.innerHTML = `
      Risk amount per trade:
      <b>${money(capital * riskPercent / 100)}</b>
      &nbsp; | &nbsp;
      Maximum allocation:
      <b>${money(capital * maxAllocationPercent / 100)}</b>
    `;
  }
}

// FILTER BUTTONS
document.querySelectorAll(".filters button").forEach(b => {
  b.onclick = () => {
    document
      .querySelectorAll(".filters button")
      .forEach(z => z.classList.remove("active"));

    b.classList.add("active");

    filter = b.dataset.f;
    render();
  };
});

// LOAD DATA
fetch("data.json?" + Date.now())
  .then(r => r.json())
  .then(d => {
    DATA = d;
    render();
  })
  .catch(() => {
    const el = document.querySelector("#updated");
    if (el) el.textContent = "Data unavailable";
  });
