"""
show_training_data.py
─────────────────────
Reads trading_agent.db, runs data_pipeline formatting logic,
and generates an interactive HTML showing every training sample
exactly as the model sees it during fine-tuning.

Usage:
    python show_training_data.py --db trading_agent.db --out viz/training_data.html
    open viz/training_data.html
"""
import json, sqlite3, argparse, textwrap
from pathlib import Path
from datetime import datetime


PRE_SYSTEM = "You are a disciplined Indian equity trader. You analyse NSE stocks using technical indicators (EMA, VWAP, support/resistance zones, HH/HL/LH/LL trend structure) and fundamental data (quarterly PAT, GNPA, NII). Your primary rule: protect capital first.\n\nOutput ONLY valid JSON:\n{\n  \"action\": \"BUY\" | \"SELL\" | \"HOLD\",\n  \"quantity\": <int>,\n  \"stop_loss\": <float>,\n  \"take_profit\": <float>,\n  \"risk_reward\": <float>,\n  \"reasoning\": \"<2-3 sentences citing specific prices and indicator values>\",\n  \"confidence\": <0.0-1.0>,\n  \"key_risk\": \"<one-line failure scenario>\"\n}"

POST_SYSTEM = "You are reviewing a completed stock trade to extract a concrete lesson for future improvement. Be specific, cite actual prices.\n\nOutput ONLY valid JSON:\n{\n  \"outcome\": \"profit\" | \"loss\" | \"breakeven\",\n  \"exit_reason_analysis\": \"<what caused the exit>\",\n  \"what_went_right\": \"<if profit: correct setup element>\",\n  \"what_went_wrong\": \"<if loss: missed signal>\",\n  \"lesson_learned\": \"<one concrete trading rule>\",\n  \"strategy_adjustment\": \"<how to change next entry>\"\n}"


def load_trades(db_path):
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM trade_log ORDER BY episode").fetchall()
    return [dict(r) for r in rows]


def build_pre_user(t, prev_trades, tech):
    """Reconstruct the user prompt exactly as given to the LLM."""
    rc = tech.get("recent_candles", [])
    candle_str = "\n".join(
        f"  {c['date']} | O:{c['open']:.2f} H:{c['high']:.2f} L:{c['low']:.2f} "
        f"C:{c['close']:.2f} V:{int(c.get('volume',0)):,}"
        for c in rc
    ) or "  (no candle data stored)"

    hist = []
    for pt in prev_trades[-4:]:
        pnl = pt.get("pnl", 0)
        hist.append(f"  Ep{pt['episode']:02d} {pt['decision_date']} | "
                    f"{pt['action']} @ {pt['entry_price']:.1f} -> {pt['exit_price']:.1f} "
                    f"[{pt['exit_reason']}] P&L {'+' if pnl>=0 else ''}{pnl:.0f} | "
                    f"Lesson: {pt.get('lesson_learned','—')}")
    hist_str = "\n".join(hist) if hist else "  (no previous trades)"

    ns = tech.get("nearest_support")
    nr = tech.get("nearest_resistance")
    return f"""=== MARKET DATA: {t['symbol']} as of {t['decision_date']} ===
Close:  {tech.get('close','—')}   VWAP: {tech.get('vwap','—')} ({'above' if tech.get('above_vwap') else 'below'} VWAP)
EMA50:  {tech.get('ema_50','—')}   EMA100: {tech.get('ema_100','—')}   EMA200: {tech.get('ema_200','—')}
Stack:  {tech.get('ema_stack','—')}   Trend: {tech.get('trend','—')}
Structure: {', '.join(tech.get('structure',[]) or ['—'])}

Nearest Support:    {ns}
Nearest Resistance: {nr}
Support Zones:    {tech.get('support_zones','—')}
Resistance Zones: {tech.get('resistance_zones','—')}

Swing Highs: {tech.get('swing_highs','—')}
Swing Lows:  {tech.get('swing_lows','—')}

Last 5 candles (Date | O H L C V):
{candle_str}

=== FUNDAMENTALS (data <= {t['decision_date']}) ===
(stored in fundamental_cache — see separate tab in dashboard)

=== TRADE HISTORY ===
{hist_str}

=== CAPITAL: {t['capital_before']:.0f} ===
Decide. Output JSON only."""


def build_pre_assistant(t):
    """The expected JSON output the model should produce."""
    return json.dumps({
        "action":      t["action"],
        "quantity":    t["quantity"],
        "stop_loss":   t["stop_loss"],
        "take_profit": t["take_profit"],
        "risk_reward": round(
            (t["take_profit"] - t["entry_price"]) / max(t["entry_price"] - t["stop_loss"], 0.01), 2
        ) if t["action"] == "BUY" and t["stop_loss"] else 0.0,
        "reasoning":   t["pre_trade_reasoning"],
        "confidence":  0.7,
        "key_risk":    f"SL at {t['stop_loss']:.2f} if trend reverses",
    }, indent=2)


def build_post_user(t):
    return f"""=== COMPLETED TRADE ===
Symbol: {t['symbol']}   Date: {t['decision_date']}
Action: {t['action']}   Qty: {t['quantity']}
Entry:  {t['entry_price']:.2f}   SL: {t['stop_loss']:.2f}   TP: {t['take_profit']:.2f}
Exit:   {t['exit_price']:.2f} on {t['exit_date'] or '—'} [{t['exit_reason']}]
P&L:    {t['pnl']:+.2f}  |  Capital: {t['capital_before']:.0f} -> {t['capital_after']:.0f}

Original reasoning: {t['pre_trade_reasoning']}

Analyse. Output JSON only."""


def build_post_assistant(t):
    outcome = "profit" if t["pnl"] > 0 else "loss" if t["pnl"] < 0 else "breakeven"
    return json.dumps({
        "outcome":            outcome,
        "exit_reason_analysis": t.get("post_trade_analysis", "—"),
        "what_went_right":    "Setup conditions were met" if outcome == "profit" else "—",
        "what_went_wrong":    t.get("what_went_wrong", "—"),
        "lesson_learned":     t.get("lesson_learned", "—"),
        "strategy_adjustment": t.get("strategy_adjustment", "—"),
    }, indent=2)


def build_samples(trades):
    """Build all training samples from the trades list."""
    samples = []
    for i, t in enumerate(trades):
        if t["action"] == "HOLD":
            continue
        tech = {}
        try:
            tech = json.loads(t.get("technicals_json") or "{}")
        except Exception:
            pass

        prev = trades[:i]

        # Pre-trade sample
        samples.append({
            "id":       f"ep{t['episode']}_pre",
            "type":     "pre_trade",
            "episode":  t["episode"],
            "symbol":   t["symbol"],
            "date":     t["decision_date"],
            "action":   t["action"],
            "pnl":      t["pnl"],
            "system":   PRE_SYSTEM,
            "user":     build_pre_user(t, prev, tech),
            "assistant": build_pre_assistant(t),
            "token_est": 800,
        })

        # Post-trade sample (only if lesson was stored)
        lesson = t.get("lesson_learned", "—")
        if lesson and lesson not in ("—", "", "Analysis unavailable (Ollama not running)"):
            samples.append({
                "id":       f"ep{t['episode']}_post",
                "type":     "post_trade",
                "episode":  t["episode"],
                "symbol":   t["symbol"],
                "date":     t["decision_date"],
                "action":   t["action"],
                "pnl":      t["pnl"],
                "system":   POST_SYSTEM,
                "user":     build_post_user(t),
                "assistant": build_post_assistant(t),
                "token_est": 350,
            })
    return samples


def build_html(trades, samples):
    samples_json = json.dumps(samples, ensure_ascii=False)
    stats = {
        "total": len(samples),
        "pre":   sum(1 for s in samples if s["type"] == "pre_trade"),
        "post":  sum(1 for s in samples if s["type"] == "post_trade"),
        "buys":  sum(1 for s in samples if s["type"] == "pre_trade" and s["action"] == "BUY"),
        "sells": sum(1 for s in samples if s["type"] == "pre_trade" and s["action"] == "SELL"),
        "wins":  sum(1 for s in samples if s["pnl"] > 0),
        "losses":sum(1 for s in samples if s["pnl"] < 0),
        "avg_tokens": int(sum(s["token_est"] for s in samples) / max(len(samples), 1)),
        "total_tokens": sum(s["token_est"] for s in samples),
    }
    stats_json = json.dumps(stats)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Training Dataset Viewer</title>
<style>
:root{{--bg:#0f1117;--card:#1a1d27;--card2:#21253a;--line:#2d3250;
      --tx:#e2e8f0;--mu:#94a3b8;--ac:#818cf8;--up:#34d399;--dn:#f87171;
      --warn:#fbbf24;--pre:#6ee7b7;--post:#93c5fd;
      --font:'JetBrains Mono','Fira Code',monospace}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font-family:'Inter',sans-serif;font-size:13px;line-height:1.6}}
.app{{max-width:1400px;margin:0 auto;padding:20px}}
/* header */
.header{{background:var(--card);border:1px solid var(--line);border-radius:12px;
         padding:20px 24px;margin-bottom:20px}}
.header h1{{font-size:20px;font-weight:800;margin-bottom:4px}}
.header .sub{{color:var(--mu);font-size:12px}}
/* stat chips */
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px}}
.stat{{background:var(--card2);border:1px solid var(--line);border-radius:8px;
       padding:8px 14px;font-size:12px;min-width:90px}}
.stat label{{display:block;color:var(--mu);font-size:10px;text-transform:uppercase;letter-spacing:.05em}}
.stat b{{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}}
/* layout */
.layout{{display:grid;grid-template-columns:280px 1fr;gap:16px;align-items:start}}
/* sidebar */
.sidebar{{background:var(--card);border:1px solid var(--line);border-radius:12px;
          overflow:hidden;position:sticky;top:20px;max-height:calc(100vh - 40px);
          display:flex;flex-direction:column}}
.sidebar-head{{padding:12px 14px;border-bottom:1px solid var(--line);
               font-size:11px;font-weight:700;text-transform:uppercase;
               letter-spacing:.06em;color:var(--mu)}}
.filter-row{{padding:10px 14px;border-bottom:1px solid var(--line);display:flex;gap:6px}}
.filter-row input{{background:var(--card2);border:1px solid var(--line);color:var(--tx);
                   padding:6px 10px;border-radius:6px;font-size:12px;width:100%;outline:none}}
.filter-row input:focus{{border-color:var(--ac)}}
.filter-btns{{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:4px;flex-wrap:wrap}}
.fbtn{{background:none;border:1px solid var(--line);color:var(--mu);padding:4px 10px;
       border-radius:5px;cursor:pointer;font-size:11px;font-weight:600}}
.fbtn.on{{background:var(--ac);border-color:var(--ac);color:#fff}}
.sample-list{{overflow-y:auto;flex:1}}
.sample-row{{padding:10px 14px;cursor:pointer;border-bottom:1px solid var(--line);
             display:flex;flex-direction:column;gap:2px;transition:background .1s}}
.sample-row:hover{{background:var(--card2)}}
.sample-row.active{{background:var(--card2);border-left:3px solid var(--ac);padding-left:11px}}
.srow-id{{font-size:10px;font-family:var(--font);color:var(--mu)}}
.srow-action{{font-size:12px;font-weight:700}}
.srow-pnl{{font-size:11px}}
/* main content */
.main{{display:flex;flex-direction:column;gap:14px}}
/* message cards */
.msg-card{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.msg-header{{padding:10px 16px;border-bottom:1px solid var(--line);
             display:flex;justify-content:space-between;align-items:center}}
.msg-header h2{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em}}
.role-badge{{font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px}}
.role-system{{background:#312e8155;color:#a5b4fc}}
.role-user{{background:#16423255;color:var(--up)}}
.role-assistant{{background:#1e3a5f55;color:var(--post)}}
.msg-body{{padding:16px;font-family:var(--font);font-size:12px;line-height:1.75;
           white-space:pre-wrap;max-height:480px;overflow-y:auto;color:#cbd5e1}}
/* json highlight */
.msg-body.json-view{{color:#e2e8f0}}
.jkey{{color:#93c5fd}}
.jstr{{color:#86efac}}
.jnum{{color:#fca5a5}}
.jbool{{color:#fbbf24}}
/* jsonl preview */
.jsonl-card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}}
.jsonl-card h2{{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
                color:var(--mu);margin-bottom:12px}}
.jsonl-box{{background:#0a0d14;border:1px solid var(--line);border-radius:8px;padding:14px;
            font-family:var(--font);font-size:11px;line-height:1.7;
            white-space:pre-wrap;overflow-x:auto;max-height:320px;overflow-y:auto;color:#94a3b8}}
.copy-btn{{background:var(--card2);border:1px solid var(--line);color:var(--mu);
           padding:4px 10px;border-radius:5px;cursor:pointer;font-size:11px;float:right}}
.copy-btn:hover{{color:var(--tx)}}
/* pills */
.pill{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700}}
.pill.buy{{background:#0d3321;color:var(--up)}}
.pill.sell{{background:#3a1515;color:var(--dn)}}
.pill.hold{{background:#2a2010;color:var(--warn)}}
.pill.pre{{background:#1a1d3a;color:var(--ac)}}
.pill.post{{background:#1a2a3a;color:var(--post)}}
.pill.profit{{background:#0d3321;color:var(--up)}}
.pill.loss{{background:#3a1515;color:var(--dn)}}
@media(max-width:900px){{.layout{{grid-template-columns:1fr}}
  .sidebar{{position:static;max-height:300px}}}}
</style>
</head>
<body>
<div class="app">

<!-- header -->
<div class="header">
  <h1>Training Dataset Viewer</h1>
  <div class="sub">Each row below is one training example — exactly what the LLM sees during fine-tuning</div>
  <div class="stats" id="statsRow"></div>
</div>

<div class="layout">
<!-- sidebar: sample list -->
<div class="sidebar">
  <div class="sidebar-head">Samples</div>
  <div class="filter-row">
    <input id="searchBox" placeholder="Search episodes, symbols..." oninput="filterSamples()">
  </div>
  <div class="filter-btns">
    <button class="fbtn on" data-f="all"  onclick="setFilter('all')">All</button>
    <button class="fbtn" data-f="pre"  onclick="setFilter('pre')">Pre-trade</button>
    <button class="fbtn" data-f="post" onclick="setFilter('post')">Post-trade</button>
    <button class="fbtn" data-f="win"  onclick="setFilter('win')">Wins</button>
    <button class="fbtn" data-f="loss" onclick="setFilter('loss')">Losses</button>
  </div>
  <div class="sample-list" id="sampleList"></div>
</div>

<!-- main: message view -->
<div class="main">
  <div id="sampleMeta" style="display:none;background:var(--card);border:1px solid var(--line);
    border-radius:12px;padding:14px 18px;display:flex;gap:12px;flex-wrap:wrap;align-items:center">
  </div>

  <div class="msg-card" id="systemCard">
    <div class="msg-header">
      <h2>System prompt <span style="color:var(--mu);font-weight:400;text-transform:none">(defines the agent's role and output schema)</span></h2>
      <span class="role-badge role-system">system</span>
    </div>
    <div class="msg-body" id="systemBody">Select a sample →</div>
  </div>

  <div class="msg-card" id="userCard">
    <div class="msg-header">
      <h2>User message <span style="color:var(--mu);font-weight:400;text-transform:none">(market data + history the model receives)</span></h2>
      <span class="role-badge role-user">user</span>
    </div>
    <div class="msg-body" id="userBody">—</div>
  </div>

  <div class="msg-card" id="assistantCard">
    <div class="msg-header">
      <h2>Expected assistant output <span style="color:var(--mu);font-weight:400;text-transform:none">(what the model must learn to produce)</span></h2>
      <span class="role-badge role-assistant">assistant</span>
    </div>
    <div class="msg-body json-view" id="assistantBody">—</div>
  </div>

  <div class="jsonl-card">
    <h2>Raw JSONL (one line in training file)
      <button class="copy-btn" onclick="copyJsonl()">Copy</button>
    </h2>
    <div class="jsonl-box" id="jsonlBox">—</div>
  </div>
</div>
</div>
</div>

<script>
const SAMPLES = {samples_json};
const STATS   = {stats_json};

let activeFilter = 'all';
let activeSample = null;

// ── stats row ─────────────────────────────────────────
function renderStats() {{
  const el = document.getElementById('statsRow');
  const items = [
    ['Total samples',  STATS.total,        '#e2e8f0'],
    ['Pre-trade',      STATS.pre,          '#818cf8'],
    ['Post-trade',     STATS.post,         '#93c5fd'],
    ['BUY samples',    STATS.buys,         '#34d399'],
    ['SELL samples',   STATS.sells,        '#f87171'],
    ['Wins',           STATS.wins,         '#34d399'],
    ['Losses',         STATS.losses,       '#f87171'],
    ['Avg tokens',     STATS.avg_tokens,   '#fbbf24'],
    ['Total tokens',   STATS.total_tokens.toLocaleString(), '#94a3b8'],
  ];
  el.innerHTML = items.map(([l,v,c])=>`
    <div class="stat">
      <label>${{l}}</label>
      <b style="color:${{c}}">${{v}}</b>
    </div>`).join('');
}}

// ── sidebar list ──────────────────────────────────────
function filterSamples() {{
  const q = document.getElementById('searchBox').value.toLowerCase();
  const list = document.getElementById('sampleList');
  let visible = SAMPLES.filter(s => {{
    if(activeFilter==='pre'  && s.type!=='pre_trade')  return false;
    if(activeFilter==='post' && s.type!=='post_trade') return false;
    if(activeFilter==='win'  && s.pnl<=0)  return false;
    if(activeFilter==='loss' && s.pnl>=0)  return false;
    if(q && !s.id.toLowerCase().includes(q) && !s.symbol.toLowerCase().includes(q)
          && !(s.action||'').toLowerCase().includes(q)) return false;
    return true;
  }});
  list.innerHTML = visible.map(s => {{
    const pnlColor = s.pnl>0?'var(--up)':s.pnl<0?'var(--dn)':'var(--mu)';
    const typePill = s.type==='pre_trade'?'pre':'post';
    const actPill  = (s.action||'hold').toLowerCase();
    return `<div class="sample-row${{s.id===activeSample?' active':''}}"
                 onclick="selectSample('${{s.id}}')">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <span class="srow-id">${{s.id}}</span>
        <span class="pill ${{typePill}}">${{s.type==='pre_trade'?'pre-trade':'post-trade'}}</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px">
        <span class="srow-action">
          <span class="pill ${{actPill}}">${{s.action}}</span>
          ${{s.symbol}} · ${{s.date}}
        </span>
        <span class="srow-pnl" style="color:${{pnlColor}}">
          ${{s.pnl>0?'+':''}}₹${{Math.abs(s.pnl).toFixed(0)}}
        </span>
      </div>
    </div>`;
  }}).join('') || '<div style="padding:20px;text-align:center;color:var(--mu);font-size:12px">No samples match</div>';
}}

function setFilter(f) {{
  activeFilter = f;
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.toggle('on',b.dataset.f===f));
  filterSamples();
}}

// ── sample viewer ─────────────────────────────────────
function selectSample(id) {{
  activeSample = id;
  const s = SAMPLES.find(x=>x.id===id);
  if(!s) return;

  // Update sidebar active state
  document.querySelectorAll('.sample-row').forEach(r=>
    r.classList.toggle('active',r.querySelector('.srow-id')?.textContent===id));

  // Meta bar
  const pnlColor = s.pnl>0?'var(--up)':s.pnl<0?'var(--dn)':'var(--mu)';
  document.getElementById('sampleMeta').style.display='flex';
  document.getElementById('sampleMeta').innerHTML = `
    <span class="pill ${{(s.action||'hold').toLowerCase()}}">${{s.action}}</span>
    <span class="pill ${{s.type==='pre_trade'?'pre':'post'}}">${{s.type.replace('_',' ')}}</span>
    <b>${{s.symbol}}</b> · ${{s.date}}
    <span style="color:var(--mu)">Ep ${{s.episode}}</span>
    <span style="color:${{pnlColor}};font-weight:700">${{s.pnl>0?'+':''}}₹${{(+s.pnl).toFixed(0)}}</span>
    <span style="color:var(--mu);font-size:11px">~${{s.token_est}} tokens</span>
  `;

  // System
  document.getElementById('systemBody').textContent = s.system;

  // User
  document.getElementById('userBody').textContent = s.user;

  // Assistant (JSON highlighted)
  document.getElementById('assistantBody').innerHTML = highlightJson(s.assistant);

  // JSONL line
  const line = JSON.stringify({{
    messages:[
      {{role:'system',  content:s.system}},
      {{role:'user',    content:s.user}},
      {{role:'assistant',content:s.assistant}},
    ]
  }});
  document.getElementById('jsonlBox').textContent = line;
}}

function highlightJson(text) {{
  return text
    .replace(/("[\w_]+")\s*:/g, '<span class="jkey">$1</span>:')
    .replace(/:\s*("(?:[^"\\\\]|\\\\.)*")/g, ': <span class="jstr">$1</span>')
    .replace(/:\s*(-?\d+\.?\d*)/g, ': <span class="jnum">$1</span>')
    .replace(/:\s*(true|false|null)/g, ': <span class="jbool">$1</span>');
}}

function copyJsonl() {{
  const text = document.getElementById('jsonlBox').textContent;
  navigator.clipboard.writeText(text).then(()=>{{
    const btn = document.querySelector('.copy-btn');
    btn.textContent='Copied!';
    setTimeout(()=>btn.textContent='Copy',2000);
  }});
}}

// ── boot ─────────────────────────────────────────────
renderStats();
filterSamples();
if(SAMPLES.length) selectSample(SAMPLES[0].id);
</script>
</body>
</html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db",  default="trading_agent.db")
    p.add_argument("--out", default="viz/training_data.html")
    args = p.parse_args()

    trades  = load_trades(args.db)
    samples = build_samples(trades)
    print(f"Trades: {len(trades)}  |  Training samples: {len(samples)}")

    html = build_html(trades, samples)
    out  = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"✓ Training data viewer → {out}  ({out.stat().st_size//1024} KB)")
    print(f"  open {out}")


if __name__ == "__main__":
    main()
