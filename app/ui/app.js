/* Trading-Agent fine-tuning — lightweight dashboard logic (no dependencies). */
const API = "/api";
const $ = (id) => document.getElementById(id);
const fmt = (v) => (v === null || v === undefined || v === "" ? "—" : v);

async function jget(path) {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(r.status + " " + path);
  return r.json();
}
async function jpost(path, body) {
  const r = await fetch(API + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.status);
  return data;
}

/* ── tabs ─────────────────────────────────────────────────────────────── */
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("panel-" + t.dataset.tab).classList.add("active");
  })
);

/* ── overview ─────────────────────────────────────────────────────────── */
const STAGES = ["datagen", "splitter", "training", "evaluation", "orchestrator", "monitoring"];
function stageCard(name, st) {
  const status = (st && st.status) || "pending";
  const pct = Math.round(((st && st.progress) || 0) * 100);
  return `<div class="stage-card">
    <div class="name">${name}</div>
    <span class="badge ${status}">${status}</span>
    <div class="progress" style="margin-top:10px"><div class="bar" style="width:${pct}%"></div></div>
    <div class="msg">${fmt(st && st.message)}</div>
  </div>`;
}
async function refreshOverview() {
  try {
    const { stages } = await jget("/overview");
    $("stageGrid").innerHTML = STAGES.map((s) => stageCard(s, stages[s])).join("");
    setConn(true);
  } catch (e) {
    setConn(false);
  }
}
function setConn(ok) {
  $("connDot").className = "dot " + (ok ? "ok" : "bad");
  $("connText").textContent = ok ? "connected" : "offline";
}

/* ── generic progress binding ─────────────────────────────────────────── */
function renderStatus(el, st) {
  el.innerHTML = [
    ["Status", st.status],
    ["Progress", Math.round((st.progress || 0) * 100) + "%"],
    ["Current", st.current],
    ["Total", st.total],
    ["Message", fmt(st.message)],
    st.error ? ["Error", st.error] : null,
  ].filter(Boolean)
   .map(([k, v]) => `<div><b>${k}</b>${fmt(v)}</div>`).join("");
}

/* ── data generation ──────────────────────────────────────────────────── */
$("dg-run").onclick = async () => {
  try {
    await jpost("/datagen/run", {
      symbol: $("dg-symbol").value, start_date: $("dg-start").value,
      episodes: +$("dg-eps").value, backend: $("dg-backend").value,
      resume: $("dg-resume").checked,
    });
  } catch (e) { alert("Datagen: " + e.message); }
};
async function refreshDatagen() {
  try {
    const st = await jget("/datagen/state");
    $("dg-bar").style.width = Math.round((st.progress || 0) * 100) + "%";
    renderStatus($("dg-status"), st);
    const { trades } = await jget("/datagen/trades?limit=25");
    $("dg-table").querySelector("tbody").innerHTML = (trades || []).map((t) => {
      const cls = t.pnl > 0 ? "pos" : t.pnl < 0 ? "neg" : "";
      return `<tr><td>${fmt(t.episode)}</td><td>${fmt(t.symbol)}</td><td>${fmt(t.decision_date)}</td>
        <td>${fmt(t.action)}</td><td>${fmt(t.entry_price)}</td><td>${fmt(t.exit_price)}</td>
        <td class="${cls}">${t.pnl != null ? t.pnl.toFixed(0) : "—"}</td><td>${fmt(t.exit_reason)}</td></tr>`;
    }).join("");
  } catch (e) {}
}

/* ── dataset viewer ───────────────────────────────────────────────────── */
async function loadRounds() {
  try {
    const { rounds } = await jget("/splitter/rounds");
    const sel = $("ds-round");
    const prev = sel.value;
    sel.innerHTML = rounds.map((r) => `<option value="${r.round.replace("round_", "")}">${r.round}</option>`).join("");
    if (prev) sel.value = prev;
    $("ds-counts").innerHTML = rounds.map((r) =>
      `<div class="stage-card"><div class="name">${r.round}</div>
       <div class="msg">train ${r.counts.train} · val ${r.counts.val} · test ${r.counts.test}</div></div>`
    ).join("") || '<p class="hint">No datasets yet — run the splitter.</p>';
  } catch (e) {}
}
$("ds-load").onclick = async () => {
  try {
    const rn = parseInt($("ds-round").value || "1", 10);
    const split = $("ds-split").value;
    const d = await jget(`/splitter/dataset?round_num=${rn}&split=${split}&limit=25`);
    $("ds-meta").innerHTML = `<div><b>Round</b>${d.round}</div><div><b>Split</b>${d.split}</div><div><b>Total</b>${d.total}</div>`;
    $("ds-rows").innerHTML = (d.rows || []).map((r) =>
      `<pre>${escapeHtml(JSON.stringify(r, null, 2))}</pre>`).join("") || '<p class="hint">Empty split.</p>';
  } catch (e) { alert("Dataset: " + e.message); }
};
function escapeHtml(s) { return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

/* ── training ─────────────────────────────────────────────────────────── */
$("tr-run").onclick = async () => {
  try {
    await jpost("/training/run", {
      round_num: +$("tr-round").value, iters: +$("tr-iters").value,
      lora_r: +$("tr-r").value, resume: $("tr-resume").checked,
    });
  } catch (e) { alert("Training: " + e.message); }
};
async function refreshTraining() {
  try {
    const st = await jget("/training/state");
    $("tr-bar").style.width = Math.round((st.progress || 0) * 100) + "%";
    renderStatus($("tr-status"), st);
    const loss = (st.history || []).filter((h) => h.event === "loss");
    drawLoss($("tr-chart"), loss);
  } catch (e) {}
}

/* ── evaluation ───────────────────────────────────────────────────────── */
$("ev-run").onclick = async () => {
  try {
    await jpost("/evaluation/run", {
      round_num: +$("ev-round").value, model_quality: $("ev-quality").checked,
    });
  } catch (e) { alert("Evaluation: " + e.message); }
};
async function refreshEvaluation() {
  try {
    const st = await jget("/evaluation/state");
    renderStatus($("ev-status"), st);
    const evs = (st.history || []).filter((h) => h.event === "eval");
    drawMetrics($("ev-chart"), evs);
    const rn = parseInt($("ev-round").value || "1", 10);
    const m = await jget(`/evaluation/metrics?round_num=${rn}`);
    const tm = (m && m.trading) || {};
    const cards = [
      ["Win rate %", tm.win_rate], ["Expectancy", tm.expectancy],
      ["Profit factor", tm.profit_factor], ["Sharpe/trade", tm.sharpe_per_trade],
      ["Max drawdown", tm.max_drawdown], ["Trades", tm.n],
    ];
    $("ev-metrics").innerHTML = cards.map(([l, v]) =>
      `<div class="metric"><div class="label">${l}</div><div class="value">${fmt(v)}</div></div>`).join("");
  } catch (e) {}
}

/* ── flywheel ─────────────────────────────────────────────────────────── */
$("fw-run").onclick = async () => {
  try {
    await jpost("/orchestrator/run", {
      symbol: $("fw-symbol").value, start_date: $("fw-start").value,
      rounds: +$("fw-rounds").value, episodes: +$("fw-eps").value,
      resume: $("fw-resume").checked,
    });
    alert("Flywheel started — track progress on the Overview board.");
  } catch (e) { alert("Flywheel: " + e.message); }
};

/* ── tiny canvas charts (no libs) ─────────────────────────────────────── */
function chartFrame(ctx, w, h) {
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#EAECEE"; ctx.lineWidth = 1;
  ctx.strokeRect(40, 10, w - 55, h - 40);
}
function drawLoss(canvas, pts) {
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  chartFrame(ctx, w, h);
  const tr = pts.filter((p) => p.train_loss != null).map((p) => ({ x: p.iter || 0, y: p.train_loss }));
  const vl = pts.filter((p) => p.val_loss != null).map((p) => ({ x: p.iter || 0, y: p.val_loss }));
  if (!tr.length && !vl.length) { emptyChart(ctx, w, h, "No loss data yet"); return; }
  const all = tr.concat(vl);
  const maxX = Math.max(1, ...all.map((p) => p.x)), maxY = Math.max(...all.map((p) => p.y)) * 1.1 || 1;
  const sx = (x) => 40 + (x / maxX) * (w - 55), sy = (y) => h - 30 - (y / maxY) * (h - 40);
  line(ctx, tr, sx, sy, "#1B90FF");
  line(ctx, vl, sx, sy, "#002A86");
  legend(ctx, w, [["train", "#1B90FF"], ["val", "#002A86"]]);
}
function drawMetrics(canvas, evs) {
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  chartFrame(ctx, w, h);
  const win = evs.map((e) => ({ x: e.round || 0, y: e.win_rate || 0 }));
  if (!win.length) { emptyChart(ctx, w, h, "No evaluation runs yet"); return; }
  const maxX = Math.max(1, ...win.map((p) => p.x)), maxY = 100;
  const sx = (x) => 40 + (x / maxX) * (w - 55), sy = (y) => h - 30 - (y / maxY) * (h - 40);
  line(ctx, win, sx, sy, "#1B90FF", true);
  legend(ctx, w, [["win rate %", "#1B90FF"]]);
}
function line(ctx, pts, sx, sy, color, dots) {
  if (!pts.length) return;
  pts.sort((a, b) => a.x - b.x);
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y))));
  ctx.stroke();
  if (dots) { ctx.fillStyle = color; pts.forEach((p) => { ctx.beginPath(); ctx.arc(sx(p.x), sy(p.y), 3, 0, 7); ctx.fill(); }); }
}
function legend(ctx, w, items) {
  ctx.font = "12px Arial"; let x = w - 160;
  items.forEach(([t, c], i) => { ctx.fillStyle = c; ctx.fillRect(x, 14 + i * 16, 10, 10); ctx.fillStyle = "#666"; ctx.fillText(t, x + 15, 23 + i * 16); });
}
function emptyChart(ctx, w, h, msg) { ctx.fillStyle = "#999"; ctx.font = "13px Arial"; ctx.fillText(msg, w / 2 - 60, h / 2); }

/* ── polling loop ─────────────────────────────────────────────────────── */
async function tick() {
  await refreshOverview();
  await refreshDatagen();
  await refreshTraining();
  await refreshEvaluation();
}
loadRounds();
tick();
setInterval(tick, 3000);
setInterval(loadRounds, 8000);
