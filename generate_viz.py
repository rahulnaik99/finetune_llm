"""
generate_viz.py
───────────────
Reads trading_agent.db and generates a self-contained interactive HTML
that shows everything the agent saw and decided.

Usage:
    python generate_viz.py --db trading_agent.db --out viz/dashboard.html
    open viz/dashboard.html
"""
import json, sqlite3, argparse
from pathlib import Path
from datetime import datetime


def load_trades(db_path: str) -> list[dict]:
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM trade_log ORDER BY episode").fetchall()
    return [dict(r) for r in rows]


def build_html(trades: list[dict]) -> str:
    trades_json = json.dumps(trades, default=str)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trading Agent — Session Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
:root{{
  --bg:#131722;--card:#1e2230;--line:#2a2e39;--tx:#d1d4dc;--mu:#93a0b8;
  --ac:#2962ff;--up:#26a69a;--dn:#ef5350;--warn:#f59e0b;
  --font:'Inter',-apple-system,sans-serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font-family:var(--font);font-size:13px;line-height:1.5}}
h1{{font-size:18px;font-weight:700;letter-spacing:-.3px}}
h2{{font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--mu);margin-bottom:10px}}
.app{{display:grid;grid-template-columns:1fr 360px;grid-template-rows:auto 1fr auto;
      min-height:100vh;gap:0}}
/* ── topbar ── */
.topbar{{grid-column:1/-1;background:var(--card);border-bottom:1px solid var(--line);
         padding:12px 20px;display:flex;gap:16px;align-items:center;flex-wrap:wrap}}
.sym{{font-size:22px;font-weight:800;letter-spacing:-.5px}}
.badge{{padding:3px 10px;border-radius:6px;font-size:11px;font-weight:700}}
.badge.profit{{background:#1a3a35;color:var(--up)}}
.badge.loss{{background:#3a1a1a;color:var(--dn)}}
.badge.hold{{background:#2a2a1a;color:var(--warn)}}
.kpi{{display:flex;flex-direction:column;gap:2px;padding:0 12px;border-left:1px solid var(--line)}}
.kpi label{{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.05em}}
.kpi b{{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums}}
.spacer{{flex:1}}
/* ── left: chart area ── */
.chart-area{{background:var(--bg);padding:16px;display:flex;flex-direction:column;gap:12px;
             border-right:1px solid var(--line)}}
/* ── episode nav ── */
.ep-nav{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.ep-nav button{{background:var(--card);border:1px solid var(--line);color:var(--tx);
                padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px;transition:all .15s}}
.ep-nav button:hover{{border-color:var(--ac);color:var(--ac)}}
.ep-nav button.active{{background:var(--ac);border-color:var(--ac);color:#fff}}
/* ── tabs ── */
.tabs{{display:flex;gap:4px}}
.tabs button{{background:none;border:none;color:var(--mu);padding:6px 14px;cursor:pointer;
              font-size:12px;border-bottom:2px solid transparent;font-weight:600}}
.tabs button.on{{color:var(--ac);border-bottom-color:var(--ac)}}
/* ── right: info panel ── */
.info-panel{{background:var(--card);padding:16px;overflow-y:auto;display:flex;flex-direction:column;gap:16px}}
.card{{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:14px}}
/* ── model input block ── */
.data-block{{font-family:monospace;font-size:11px;line-height:1.7;color:var(--mu);
             white-space:pre-wrap;max-height:260px;overflow-y:auto}}
.data-block .hi{{color:var(--tx)}}
.data-block .up{{color:var(--up)}}
.data-block .dn{{color:var(--dn)}}
.data-block .wa{{color:var(--warn)}}
/* ── trade decision box ── */
.decision-box{{border-radius:10px;padding:14px;border:1px solid var(--line)}}
.decision-box.buy{{background:#0d2320;border-color:#26a69a44}}
.decision-box.sell{{background:#230d0d;border-color:#ef535044}}
.decision-box.hold{{background:#231f0d;border-color:#f59e0b44}}
.price-row{{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}}
.price-tag{{padding:4px 10px;border-radius:6px;font-size:12px;font-weight:700}}
.price-tag.entry{{background:#2962ff22;color:#2962ff}}
.price-tag.sl{{background:#ef535022;color:var(--dn)}}
.price-tag.tp{{background:#26a69a22;color:var(--up)}}
.price-tag.exit{{background:#ffffff15;color:var(--tx)}}
/* ── reasoning ── */
.reasoning{{font-size:12px;line-height:1.65;color:var(--mu);margin-top:8px;
            border-left:2px solid var(--ac);padding-left:10px}}
/* ── lesson ── */
.lesson{{font-size:12px;line-height:1.65;margin-top:8px;padding:10px;
         border-radius:8px;background:#ffffff08}}
.lesson.good{{border-left:3px solid var(--up)}}
.lesson.bad{{border-left:3px solid var(--dn)}}
/* ── bottom: capital + history ── */
.bottom-panel{{grid-column:1/-1;background:var(--card);border-top:1px solid var(--line);padding:16px}}
.history-table{{width:100%;border-collapse:collapse;font-size:12px}}
.history-table th{{color:var(--mu);font-size:10px;text-transform:uppercase;letter-spacing:.05em;
                   padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;
                   background:var(--card);position:sticky;top:0}}
.history-table td{{padding:6px 10px;border-bottom:1px solid var(--line)}}
.history-table tr:hover td{{background:var(--line)}}
.tw{{max-height:220px;overflow-y:auto;border:1px solid var(--line);border-radius:8px}}
@media(max-width:900px){{.app{{grid-template-columns:1fr}}.info-panel{{border-top:1px solid var(--line)}}}}
</style>
</head>
<body>
<div class="app">

<!-- topbar -->
<div class="topbar">
  <div>
    <div class="sym" id="sym">—</div>
    <div style="font-size:11px;color:var(--mu)" id="subline">Select an episode to begin</div>
  </div>
  <div class="kpi">
    <label>Capital</label>
    <b id="kpiCap">—</b>
  </div>
  <div class="kpi">
    <label>P&amp;L this trade</label>
    <b id="kpiPnl">—</b>
  </div>
  <div class="kpi">
    <label>Win rate</label>
    <b id="kpiWr">—</b>
  </div>
  <div class="kpi">
    <label>Max DD</label>
    <b id="kpiDD">—</b>
  </div>
  <div class="spacer"></div>
  <div id="actionBadge" class="badge">—</div>
</div>

<!-- chart area -->
<div class="chart-area">
  <!-- episode nav -->
  <div>
    <h2>Episodes</h2>
    <div class="ep-nav" id="epNav"></div>
  </div>

  <!-- TF tabs -->
  <div>
    <div class="tabs" id="tfTabs">
      <button class="on" onclick="switchTF('1D')">1D Chart</button>
      <button onclick="switchTF('Capital')">Capital Curve</button>
    </div>
  </div>

  <!-- chart -->
  <div id="mainChart" style="flex:1;min-height:420px"></div>

  <!-- what model sees: technical summary -->
  <div class="card">
    <h2>Technical snapshot (what the model sees)</h2>
    <div class="data-block" id="techBlock">—</div>
  </div>
</div>

<!-- right info panel -->
<div class="info-panel">

  <div class="card">
    <h2>Model decision</h2>
    <div id="decisionBox" class="decision-box hold">
      <div style="font-size:18px;font-weight:800" id="actionLabel">HOLD</div>
      <div class="price-row" id="priceTags"></div>
      <div class="reasoning" id="reasoning">Select an episode →</div>
    </div>
  </div>

  <div class="card">
    <h2>Outcome &amp; lesson</h2>
    <div id="outcomeBox">
      <div style="font-size:12px;color:var(--mu)" id="exitDetail">—</div>
      <div class="lesson" id="lessonBox">—</div>
      <div style="font-size:11px;color:var(--mu);margin-top:8px" id="wrongBox">—</div>
    </div>
  </div>

  <div class="card">
    <h2>Fundamentals (as of episode date)</h2>
    <div class="data-block" id="fundBlock">—</div>
  </div>

  <div class="card">
    <h2>Position sizing logic</h2>
    <div class="data-block" id="sizeBlock">—</div>
  </div>

</div>

<!-- bottom panel -->
<div class="bottom-panel">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
    <div>
      <h2 style="margin-bottom:8px">All trades</h2>
      <div class="tw">
        <table class="history-table">
          <thead>
            <tr>
              <th>#</th><th>Date</th><th>Act</th><th>Entry</th>
              <th>SL</th><th>TP</th><th>Exit</th><th>Reason</th>
              <th>P&amp;L</th><th>Capital</th><th>Lesson</th>
            </tr>
          </thead>
          <tbody id="histTbody"></tbody>
        </table>
      </div>
    </div>
    <div>
      <h2 style="margin-bottom:8px">What the model predicted vs what happened</h2>
      <div id="predVsActual" style="font-size:12px;color:var(--mu);line-height:1.8">
        Select an episode to see the prediction accuracy.
      </div>
    </div>
  </div>
</div>

</div><!-- /app -->

<script>
// ═══════════════════════════════════════════════════════
//  DATA
// ═══════════════════════════════════════════════════════
const TRADES = {trades_json};
const INITIAL_CAP = 10000;

// ── helpers ─────────────────────────────────────────────
const esc = s => String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const fmt = (n,d=2) => Number.isFinite(+n) ? (+n).toLocaleString('en-IN',{{maximumFractionDigits:d}}) : '—';
const sgn = n => n>=0?'+':'-';
const pnlColor = n => n>0?'var(--up)':n<0?'var(--dn)':'var(--mu)';

let currentEp = 1;
let currentTF = '1D';

// ── compute running stats ────────────────────────────────
function stats() {{
  let peak = INITIAL_CAP, mdd = 0;
  for(const t of TRADES) {{
    peak = Math.max(peak, t.capital_after);
    mdd  = Math.min(mdd, (t.capital_after - peak)/peak*100);
  }}
  const wins   = TRADES.filter(t=>t.pnl>0).length;
  const losses = TRADES.filter(t=>t.pnl<0).length;
  const total  = TRADES.filter(t=>t.action!=='HOLD').length;
  return {{ mdd, wr: total>0? wins/total : 0, wins, losses }};
}}

// ═══════════════════════════════════════════════════════
//  EPISODE NAVIGATION
// ═══════════════════════════════════════════════════════
function buildEpNav() {{
  const container = document.getElementById('epNav');
  container.innerHTML = TRADES.map(t => {{
    const cls = t.pnl>0?'profit':t.pnl<0?'loss':t.action==='HOLD'?'hold':'';
    return `<button class="episode-btn ${{cls}}" data-ep="${{t.episode}}"
              onclick="selectEpisode(${{t.episode}})"
              title="Ep ${{t.episode}}: ${{t.action}} ${{t.pnl>0?'+':''}}${{(+t.pnl).toFixed(0)}}">
              ${{t.episode}}
            </button>`;
  }}).join('');
}}

function selectEpisode(ep) {{
  currentEp = ep;
  document.querySelectorAll('.episode-btn').forEach(b => {{
    b.classList.toggle('active', +b.dataset.ep === ep);
  }});
  const t = TRADES.find(x=>x.episode===ep);
  if(t) renderEpisode(t);
}}

// ═══════════════════════════════════════════════════════
//  RENDER ONE EPISODE
// ═══════════════════════════════════════════════════════
function renderEpisode(t) {{
  // ── topbar ─────────────────────────────────────────────
  document.getElementById('sym').textContent = t.symbol;
  document.getElementById('subline').textContent =
    `Episode ${{t.episode}} · ${{t.decision_date}} · ${{t.exit_reason||'—'}}`;

  const cap = t.capital_after;
  const pnl = t.pnl;
  document.getElementById('kpiCap').textContent = `₹${{fmt(cap,0)}}`;
  document.getElementById('kpiPnl').innerHTML =
    `<span style="color:${{pnlColor(pnl)}}">${{sgn(pnl)}}₹${{fmt(Math.abs(pnl),0)}}</span>`;

  const s = stats();
  document.getElementById('kpiWr').textContent = `${{(s.wr*100).toFixed(0)}}%`;
  document.getElementById('kpiDD').innerHTML =
    `<span style="color:${{s.mdd<-10?'var(--dn)':'var(--mu)}}">${{s.mdd.toFixed(1)}}%</span>`;

  const actionCls = t.action==='BUY'?'profit':t.action==='SELL'?'loss':'hold';
  const badge = document.getElementById('actionBadge');
  badge.textContent = t.action;
  badge.className = `badge ${{actionCls}}`;

  // ── chart ───────────────────────────────────────────────
  if(currentTF==='1D') renderCandleChart(t);
  else renderCapitalChart();

  // ── technical block ─────────────────────────────────────
  const tech = safeJson(t.technicals_json);
  document.getElementById('techBlock').innerHTML = buildTechBlock(tech, t);

  // ── decision box ────────────────────────────────────────
  const box = document.getElementById('decisionBox');
  box.className = `decision-box ${{t.action.toLowerCase()}}`;
  document.getElementById('actionLabel').textContent = t.action;
  document.getElementById('priceTags').innerHTML = t.action !== 'HOLD' ? `
    <span class="price-tag entry">Entry ₹${{fmt(t.entry_price)}}</span>
    <span class="price-tag sl">SL ₹${{fmt(t.stop_loss)}}</span>
    <span class="price-tag tp">TP ₹${{fmt(t.take_profit)}}</span>
    <span class="price-tag exit">Exit ₹${{fmt(t.exit_price)}} [${{t.exit_reason||'—'}}]</span>
  ` : '<span style="color:var(--mu);font-size:12px">No position taken</span>';

  document.getElementById('reasoning').textContent = t.pre_trade_reasoning || '—';

  // ── outcome ─────────────────────────────────────────────
  const outcome = pnl>0?'PROFIT':pnl<0?'LOSS':'FLAT';
  document.getElementById('exitDetail').innerHTML =
    `<span style="color:${{pnlColor(pnl)}};font-weight:700">${{outcome}}</span> · `+
    `Exit via <b>${{t.exit_reason||'—'}}</b> on ${{t.exit_date||'—'}} @ ₹${{fmt(t.exit_price)}} · `+
    `P&L <span style="color:${{pnlColor(pnl)}};font-weight:700">${{sgn(pnl)}}₹${{fmt(Math.abs(pnl),0)}}</span>`;

  const lessonBox = document.getElementById('lessonBox');
  lessonBox.textContent = t.lesson_learned || '—';
  lessonBox.className = `lesson ${{pnl>=0?'good':'bad'}}`;

  document.getElementById('wrongBox').innerHTML =
    t.what_went_wrong && t.what_went_wrong !== '—'
    ? `⚠ What went wrong: ${{esc(t.what_went_wrong)}}`
    : t.pnl > 0 ? '✓ Trade worked as expected' : '';

  // ── fundamentals ────────────────────────────────────────
  document.getElementById('fundBlock').innerHTML = buildFundBlock(tech, t);

  // ── position sizing ──────────────────────────────────────
  document.getElementById('sizeBlock').innerHTML = buildSizeBlock(t);

  // ── pred vs actual ───────────────────────────────────────
  document.getElementById('predVsActual').innerHTML = buildPredVsActual(t);
}}

// ═══════════════════════════════════════════════════════
//  BLOCK BUILDERS
// ═══════════════════════════════════════════════════════
function buildTechBlock(tech, t) {{
  if(!tech.close) return '<span style="color:var(--dn)">No technical data stored for this episode.</span>';
  const upDn = v => v ? `<span class="up">↑ above</span>` : `<span class="dn">↓ below</span>`;
  const trend_c = tech.trend==='uptrend'?'up':tech.trend==='downtrend'?'dn':'wa';
  return `
<span class="hi">Close:</span>  ₹${{tech.close}}   <span class="hi">VWAP:</span> ₹${{tech.vwap}} (${{upDn(tech.above_vwap)}} VWAP)
<span class="hi">EMA50:</span>  ₹${{tech.ema_50}}   <span class="hi">EMA100:</span> ₹${{tech.ema_100}}   <span class="hi">EMA200:</span> ₹${{tech.ema_200}}
<span class="hi">Stack:</span>  ${{tech.ema_stack||'—'}}
<span class="hi">Trend:</span>  <span class="${{trend_c}}">${{tech.trend||'—'}}</span>   <span class="hi">Structure:</span> ${{(tech.structure||[]).join(', ')||'—'}}

<span class="hi">Nearest Support:</span>    ${{JSON.stringify(tech.nearest_support)||'—'}}
<span class="hi">Nearest Resistance:</span> ${{JSON.stringify(tech.nearest_resistance)||'—'}}

<span class="hi">Swing Highs:</span> ${{JSON.stringify(tech.swing_highs||[])}}
<span class="hi">Swing Lows:</span>  ${{JSON.stringify(tech.swing_lows||[])}}`;
}}

function buildFundBlock(tech, t) {{
  return `<span style="color:var(--mu);font-size:11px">
Stored in fundamental_cache (quarterly, fetched once per quarter).
Retrieved for: ${{t.symbol}} as of ${{t.decision_date}}

To see full data: check trading_agent.db → fundamental_cache
WHERE symbol='${{t.symbol}}'
</span>`;
}}

function buildSizeBlock(t) {{
  if(t.action==='HOLD') return '<span style="color:var(--mu)">No position — held cash</span>';
  const risk = t.entry_price - t.stop_loss;
  const reward = t.take_profit - t.entry_price;
  const rr = risk>0 ? (reward/risk).toFixed(2) : '—';
  const riskRs = (risk * t.quantity).toFixed(0);
  const potGain = (reward * t.quantity).toFixed(0);
  const riskPct = ((risk * t.quantity) / t.capital_before * 100).toFixed(2);
  return `
<span class="hi">Quantity:</span>    ${{t.quantity}} shares
<span class="hi">Entry:</span>       ₹${{fmt(t.entry_price)}}
<span class="hi">Stop Loss:</span>   ₹${{fmt(t.stop_loss)}}   (<span class="dn">-₹${{fmt(risk,2)}} / share</span>)
<span class="hi">Take Profit:</span> ₹${{fmt(t.take_profit)}}  (<span class="up">+₹${{fmt(reward,2)}} / share</span>)

<span class="hi">Risk per trade:</span>  ₹${{riskRs}} (${{riskPct}}% of capital)
<span class="hi">Potential gain:</span> ₹${{potGain}}
<span class="hi">Risk:Reward:</span>    1 : ${{rr}}`;
}}

function buildPredVsActual(t) {{
  if(t.action==='HOLD')
    return '<span style="color:var(--mu)">Model chose HOLD — no prediction to evaluate</span>';

  const predicted = t.action==='BUY' ? 'price goes UP' : 'price goes DOWN';
  const actual = t.pnl>0 ? (t.action==='BUY'?'price went UP ✓':'price went DOWN ✓')
               : t.pnl<0 ? (t.action==='BUY'?'price went DOWN ✗':'price went UP ✗')
               : 'time exit (neither SL nor TP hit)';

  const exitExplain = {{
    'SL_HIT':   `Model set SL at ₹${{fmt(t.stop_loss)}}. Price touched it on ${{t.exit_date||'?'}}.`,
    'TP_HIT':   `Model set TP at ₹${{fmt(t.take_profit)}}. Price reached it on ${{t.exit_date||'?'}}.`,
    'TIME_EXIT': `Neither SL nor TP hit within ${{t.bars_held||'?'}} bars. Exited at close ₹${{fmt(t.exit_price)}}.`,
    'HOLD':     'Model chose not to trade.',
  }}[t.exit_reason] || t.exit_reason;

  const outcomeColor = t.pnl>0?'var(--up)':t.pnl<0?'var(--dn)':'var(--mu)';
  return `
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
      <div>
        <div style="color:var(--mu);font-size:10px;text-transform:uppercase;letter-spacing:.05em">Predicted</div>
        <div style="font-size:14px;font-weight:700;margin-top:3px">${{predicted}}</div>
        <div style="font-size:11px;color:var(--mu);margin-top:4px">
          ${{t.action}} ${{t.quantity}} @ ₹${{fmt(t.entry_price)}}
          with SL ₹${{fmt(t.stop_loss)}} / TP ₹${{fmt(t.take_profit)}}
        </div>
      </div>
      <div>
        <div style="color:var(--mu);font-size:10px;text-transform:uppercase;letter-spacing:.05em">Actual</div>
        <div style="font-size:14px;font-weight:700;margin-top:3px;color:${{outcomeColor}}">${{actual}}</div>
        <div style="font-size:11px;color:var(--mu);margin-top:4px">${{exitExplain}}</div>
      </div>
    </div>
    <div style="margin-top:12px;padding:10px;background:var(--line);border-radius:8px;font-size:11px">
      <b>Post-mortem:</b> ${{esc(t.post_trade_analysis||'—')}}
    </div>`;
}}

// ═══════════════════════════════════════════════════════
//  CHARTS
// ═══════════════════════════════════════════════════════
function switchTF(tf) {{
  currentTF = tf;
  document.querySelectorAll('#tfTabs button').forEach((b,i) => {{
    b.classList.toggle('on', b.textContent===tf);
  }});
  const t = TRADES.find(x=>x.episode===currentEp);
  if(!t) return;
  if(tf==='1D') renderCandleChart(t);
  else renderCapitalChart();
}}

function renderCandleChart(t) {{
  const tech = safeJson(t.technicals_json);
  const rc = tech.recent_candles || [];
  if(!rc.length) {{
    Plotly.purge('mainChart');
    document.getElementById('mainChart').innerHTML =
      '<div style="padding:40px;text-align:center;color:var(--mu)">No candle data stored for this episode.</div>';
    return;
  }}

  const dates = rc.map(c=>c.date);
  const opens  = rc.map(c=>c.open);
  const highs  = rc.map(c=>c.high);
  const lows   = rc.map(c=>c.low);
  const closes = rc.map(c=>c.close);
  const vols   = rc.map(c=>c.volume);

  const candles = {{
    type:'candlestick', x:dates,
    open:opens, high:highs, low:lows, close:closes,
    name:'Price',
    increasing:{{line:{{color:'#26a69a'}},fillcolor:'#26a69a'}},
    decreasing:{{line:{{color:'#ef5350'}},fillcolor:'#ef5350'}},
  }};

  const ema50 = ema(closes, 50);
  const ema200= ema(closes, 200);

  const traces = [candles,
    {{type:'scatter',x:dates,y:ema50, name:'EMA50', line:{{color:'#f39c12',width:1.5}}, yaxis:'y'}},
    {{type:'scatter',x:dates,y:ema200,name:'EMA200',line:{{color:'#9b59b6',width:1.5}}, yaxis:'y'}},
    {{type:'bar',    x:dates,y:vols,  name:'Volume',marker:{{color:closes.map((c,i)=>c>=opens[i]?'#26a69a55':'#ef535055')}},
      yaxis:'y2', showlegend:false}},
  ];

  // entry / SL / TP lines
  if(t.action !== 'HOLD' && t.entry_price) {{
    const lastDate = dates[dates.length-1];
    traces.push(
      {{type:'scatter',x:[dates[0],lastDate],y:[t.entry_price,t.entry_price],
        mode:'lines',name:'Entry',line:{{color:'#2962ff',width:1.5,dash:'dash'}}}},
      {{type:'scatter',x:[dates[0],lastDate],y:[t.stop_loss,t.stop_loss],
        mode:'lines',name:'SL',line:{{color:'#ef5350',width:1.2,dash:'dot'}}}},
      {{type:'scatter',x:[dates[0],lastDate],y:[t.take_profit,t.take_profit],
        mode:'lines',name:'TP',line:{{color:'#26a69a',width:1.2,dash:'dot'}}}},
    );
  }}

  const layout = {{
    paper_bgcolor:'#131722', plot_bgcolor:'#131722',
    font:{{color:'#93a0b8',size:11}},
    xaxis:{{gridcolor:'#2a2e39',rangeslider:{{visible:false}}}},
    yaxis:{{gridcolor:'#2a2e39',domain:[0.25,1]}},
    yaxis2:{{gridcolor:'#2a2e39',domain:[0,0.2],showticklabels:false}},
    legend:{{bgcolor:'#1e223099',font:{{size:10}}}},
    margin:{{l:55,r:20,t:20,b:40}},
    title:{{text:`${{t.symbol}} — last ${{rc.length}} daily candles (Ep ${{t.episode}})`,
            font:{{color:'#d1d4dc',size:12}},x:0.01}},
  }};

  Plotly.react('mainChart', traces, layout, {{responsive:true,displayModeBar:false}});
}}

function renderCapitalChart() {{
  const caps   = [INITIAL_CAP, ...TRADES.map(t=>t.capital_after)];
  const labels = ['Start', ...TRADES.map(t=>`Ep ${{t.episode}}`)];
  const colors = caps.map((c,i)=>i===0?'#93a0b8':c>caps[i-1]?'#26a69a':'#ef5350');

  const trace = {{
    type:'scatter', mode:'lines+markers',
    x:labels, y:caps,
    line:{{color:'#7e9bff',width:2}},
    marker:{{color:colors,size:8}},
    fill:'tozeroy', fillcolor:'#7e9bff11',
    name:'Capital',
  }};
  const layout = {{
    paper_bgcolor:'#131722', plot_bgcolor:'#131722',
    font:{{color:'#93a0b8',size:11}},
    xaxis:{{gridcolor:'#2a2e39',tickangle:-30}},
    yaxis:{{gridcolor:'#2a2e39'}},
    shapes:[
      {{type:'line',x0:0,x1:1,xref:'paper',y0:INITIAL_CAP,y1:INITIAL_CAP,
        line:{{color:'#888',dash:'dash',width:1}}}},
      {{type:'line',x0:0,x1:1,xref:'paper',y0:INITIAL_CAP*0.2,y1:INITIAL_CAP*0.2,
        line:{{color:'#ef5350',dash:'dot',width:1}}}},
    ],
    annotations:[
      {{xref:'paper',yref:'y',x:1.01,y:INITIAL_CAP,text:'Start',showarrow:false,font:{{size:10,color:'#888'}}}},
      {{xref:'paper',yref:'y',x:1.01,y:INITIAL_CAP*0.2,text:'Death',showarrow:false,font:{{size:10,color:'#ef5350'}}}},
    ],
    margin:{{l:60,r:70,t:30,b:70}},
    title:{{text:'Capital Curve across all episodes',font:{{color:'#d1d4dc',size:12}},x:0.01}},
  }};
  Plotly.react('mainChart',[trace],layout,{{responsive:true,displayModeBar:false}});
}}

// ── EMA compute (client-side) ─────────────────────────
function ema(data, period) {{
  const k = 2/(period+1);
  const result = [];
  let prev = null;
  for(const v of data) {{
    if(prev===null){{ prev=v; result.push(v); continue; }}
    prev = v*k + prev*(1-k);
    result.push(+prev.toFixed(3));
  }}
  return result;
}}

// ═══════════════════════════════════════════════════════
//  HISTORY TABLE
// ═══════════════════════════════════════════════════════
function buildHistoryTable() {{
  const tbody = document.getElementById('histTbody');
  tbody.innerHTML = TRADES.map(t => {{
    const pnl = t.pnl;
    const bg = pnl>0?'#0d2320':pnl<0?'#230d0d':'';
    const pnlHtml = `<b style="color:${{pnlColor(pnl)}}">${{sgn(pnl)}}₹${{fmt(Math.abs(pnl),0)}}</b>`;
    return `<tr style="background:${{bg}};cursor:pointer" onclick="selectEpisode(${{t.episode}})">
      <td>${{t.episode}}</td>
      <td>${{t.decision_date}}</td>
      <td><b style="color:${{t.action==='BUY'?'var(--up)':t.action==='SELL'?'var(--dn)':'var(--mu)}}">${{t.action}}</b></td>
      <td>₹${{fmt(t.entry_price)}}</td>
      <td style="color:var(--dn)">₹${{fmt(t.stop_loss)}}</td>
      <td style="color:var(--up)">₹${{fmt(t.take_profit)}}</td>
      <td>₹${{fmt(t.exit_price)}}</td>
      <td style="font-size:10px">${{t.exit_reason||'—'}}</td>
      <td>${{pnlHtml}}</td>
      <td>₹${{fmt(t.capital_after,0)}}</td>
      <td style="font-size:10px;max-width:160px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis"
          title="${{esc(t.lesson_learned)}}">${{esc((t.lesson_learned||'—').slice(0,50))}}</td>
    </tr>`;
  }}).join('');
}}

// ═══════════════════════════════════════════════════════
//  UTILS
// ═══════════════════════════════════════════════════════
function safeJson(s) {{
  try{{ return JSON.parse(s||'{{}}'); }}catch{{return {{}};}}
}}

// ═══════════════════════════════════════════════════════
//  BOOT
// ═══════════════════════════════════════════════════════
(function() {{
  if(!TRADES.length) {{
    document.getElementById('sym').textContent = 'No data';
    document.getElementById('subline').textContent = 'Run the simulation first to generate episodes';
    return;
  }}
  buildEpNav();
  buildHistoryTable();
  renderCapitalChart();   // start with capital curve
  currentTF = 'Capital';
  document.querySelectorAll('#tfTabs button').forEach(b=>b.classList.toggle('on',b.textContent==='Capital'));
  // Auto-select last episode
  const last = TRADES[TRADES.length-1];
  selectEpisode(last.episode);
}})();
</script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db",  default="trading_agent.db")
    p.add_argument("--out", default="viz/dashboard.html")
    args = p.parse_args()

    trades = load_trades(args.db)
    print(f"Loaded {len(trades)} trades")

    html = build_html(trades)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"✓ Dashboard → {out}  ({out.stat().st_size//1024} KB)")
    print(f"  open {out}")


if __name__ == "__main__":
    main()
