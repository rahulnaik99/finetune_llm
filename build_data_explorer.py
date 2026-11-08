"""
build_data_explorer.py
──────────────────────
Generate a single self-contained HTML page to browse the training data by hand.
Each row = one training record (input -> output). Click any column header to
sort; type in the filter box to search; click a row to expand the full
INPUT (prompt) and OUTPUT (decision JSON).

Data sources (pick one):
  --db trading_agent.db                 # richest: rebuilds pre+post examples with meta
  --jsonl data/rounds/round_01/trading_train.jsonl [more.jsonl ...]
  --demo                                # synthetic sample so you can see it immediately

Output:
  data_explorer.html  (open it in any browser — no server, fully offline)

Examples:
  python build_data_explorer.py --db trading_agent.db
  python build_data_explorer.py --jsonl data/rounds/round_01/trading_*.jsonl
  python build_data_explorer.py --demo
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path


def _parse_out(assistant_text: str) -> dict:
    try:
        return json.loads(assistant_text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", assistant_text or "", re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


def _record_from_example(ex: dict, split: str) -> dict:
    msgs = ex.get("messages", [])
    usr = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    meta = ex.get("_meta", {}) or {}
    o = _parse_out(asst)
    return {
        "split": split,
        "type": meta.get("type", ""),
        "symbol": meta.get("symbol", ""),
        "date": meta.get("date", ""),
        "action": o.get("action", o.get("outcome", "")),
        "conf": o.get("confidence", ""),
        "rr": o.get("risk_reward", ""),
        "pnl": meta.get("pnl", ""),
        "fund": meta.get("fund_completeness", ""),
        "exit_reason": meta.get("exit_reason", ""),
        "input": usr,
        "output": json.dumps(o, indent=2) if o else asst,
    }


def from_db(db_path: str) -> list[dict]:
    import data_pipeline as dp
    trades = dp.load_trades(db_path)
    recs = []
    for i, t in enumerate(trades):
        pre = dp.build_pre_trade_example(t, trades[:i])
        post = dp.build_post_trade_example(t)
        if pre:  recs.append(_record_from_example(pre, "-"))
        if post: recs.append(_record_from_example(post, "-"))
    return recs


def from_jsonl(paths: list[str]) -> list[dict]:
    recs = []
    for pat in paths:
        for path in glob.glob(pat):
            split = Path(path).stem.split("_")[-1]  # ..._train/_val/_test
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                recs.append(_record_from_example(json.loads(line), split))
    return recs


def demo_records() -> list[dict]:
    import random
    random.seed(7)
    recs = []
    for i in range(14):
        act = random.choice(["BUY", "SELL", "HOLD"])
        pnl = 0 if act == "HOLD" else round(random.uniform(-300, 400), 0)
        inp = (f"[CANBK 2024-01-{i+1:02d}]\n"
               f"HEADLINE: bias:{'bearish' if act=='SELL' else 'bullish'} | trend:downtrend | VIX:NORMAL\n"
               f"REL: vsEMA200:{random.uniform(-4,4):+.1f}% | vsVWAP:{random.uniform(-2,2):+.1f}% | 5dPos:{random.randint(10,90)}pct\n"
               f"EMA: 50@118.9 < 100@120.1 < 200@122.5 [BEARISH]\n"
               f"LEVELS: PDH 120.1(rejected) | PDL 118.0(held)\n"
               f"ZONES:\n  1h DZ:115.0-117.5 fresh(-2.0%)\nCAP:10000")
        out = {"action": act,
               "stop_loss": 116.5 if act != "HOLD" else 0,
               "take_profit": 123.0 if act != "HOLD" else 0,
               "risk_reward": round(random.uniform(1.5, 2.5), 2) if act != "HOLD" else 0,
               "reasoning": "EMA bearish; price rejected PDH; fresh 1h demand below.",
               "confidence": round(random.uniform(0.4, 0.8), 2),
               "features_used": ["ema_stack", "pdh_state", "zone.1h.demand"]}
        recs.append({
            "split": random.choice(["train", "val", "test"]),
            "type": "pre_trade", "symbol": "CANBK", "date": f"2024-01-{i+1:02d}",
            "action": act, "conf": out["confidence"], "rr": out["risk_reward"],
            "pnl": pnl, "fund": __import__("random").choice(["fresh","stale","none"]), "exit_reason": random.choice(["take_profit", "stop_loss", "time_exit", ""]),
            "input": inp, "output": json.dumps(out, indent=2),
        })
    return recs


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training Data Explorer</title>
<style>
  :root{--bg:#f7f8fa;--card:#fff;--line:#e3e6ea;--ink:#1a1f26;--muted:#6b7280;--accent:#0a6ed1;--pos:#157a3f;--neg:#b42318}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  header{padding:14px 18px;background:var(--card);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}
  h1{font-size:16px;margin:0 0 8px}
  .controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  input[type=search]{padding:7px 10px;border:1px solid var(--line);border-radius:8px;min-width:260px;font-size:13px}
  .pill{font-size:12px;color:var(--muted)}
  .wrap{padding:12px 18px}
  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
  th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
  th{background:#eef1f5;cursor:pointer;user-select:none;font-weight:600}
  th:hover{background:#e6eaf0}
  th .arr{color:var(--accent);font-size:11px}
  tr.rowtop{cursor:pointer}
  tr.rowtop:hover{background:#f2f6fc}
  td.num{text-align:right;font-variant-numeric:tabular-nums}
  .badge{padding:1px 7px;border-radius:999px;font-size:12px;font-weight:600}
  .BUY{background:#e7f5ec;color:var(--pos)} .SELL{background:#fdecea;color:var(--neg)} .HOLD{background:#eef1f5;color:#555}
  .pos{color:var(--pos)} .neg{color:var(--neg)}
  .detail td{background:#fbfcfe;white-space:normal}
  .io{display:flex;gap:16px;flex-wrap:wrap}
  .io > div{flex:1 1 360px;min-width:300px}
  .io h4{margin:0 0 6px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
  pre{margin:0;padding:10px;background:#0f141a;color:#e6edf3;border-radius:8px;overflow:auto;max-height:360px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
  .hidden{display:none}
</style></head>
<body>
<header>
  <h1>Training Data Explorer &nbsp;<span class="pill" id="count"></span></h1>
  <div class="controls">
    <input id="q" type="search" placeholder="filter - searches every column + input/output...">
    <span class="pill">click a column header to sort &middot; click a row to expand input/output</span>
  </div>
</header>
<div class="wrap">
  <table id="t"><thead><tr id="hr"></tr></thead><tbody id="tb"></tbody></table>
</div>
<script>
const DATA = __DATA__;
const COLS = [
  {k:"split",t:"split"},{k:"type",t:"type"},{k:"symbol",t:"symbol"},{k:"date",t:"date"},
  {k:"action",t:"action"},{k:"conf",t:"conf",n:1},{k:"rr",t:"R:R",n:1},
  {k:"pnl",t:"P&L",n:1},{k:"fund",t:"fund"},{k:"exit_reason",t:"exit"}
];
let sortK=null, sortDir=1, rows=DATA.map((d,i)=>({...d,_i:i}));
const hr=document.getElementById("hr"), tb=document.getElementById("tb"), q=document.getElementById("q");
COLS.forEach(c=>{const th=document.createElement("th");th.textContent=c.t;th.dataset.k=c.k;
  th.onclick=()=>{sortDir=(sortK===c.k)?-sortDir:1;sortK=c.k;render();};hr.appendChild(th);});
function esc(s){return (s==null?"":String(s)).replace(/[&<>]/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));}
function num(v){const f=parseFloat(v);return isNaN(f)?null:f;}
function render(){
  let r=rows.slice();
  const term=q.value.trim().toLowerCase();
  if(term) r=r.filter(d=>COLS.some(c=>String(d[c.k]==null?"":d[c.k]).toLowerCase().includes(term))
      || (d.input||"").toLowerCase().includes(term) || (d.output||"").toLowerCase().includes(term));
  if(sortK){const col=COLS.find(c=>c.k===sortK);
    r.sort((a,b)=>{let x=a[sortK],y=b[sortK];
      if(col&&col.n){x=num(x);y=num(y);x=(x==null?-1e18:x);y=(y==null?-1e18:y);return (x-y)*sortDir;}
      return String(x==null?"":x).localeCompare(String(y==null?"":y))*sortDir;});}
  document.querySelectorAll("th").forEach(th=>{const a=th.querySelector(".arr"); if(a) a.remove();});
  if(sortK){const th=[...document.querySelectorAll("th")].find(t=>t.dataset.k===sortK);
    const s=document.createElement("span");s.className="arr";s.textContent=sortDir>0?" \\u25B2":" \\u25BC";th.appendChild(s);}
  tb.innerHTML="";
  r.forEach(d=>{
    const tr=document.createElement("tr");tr.className="rowtop";
    tr.innerHTML=COLS.map(c=>{
      let v=d[c.k];
      if(c.k==="action"&&v) return `<td><span class="badge ${esc(v)}">${esc(v)}</span></td>`;
      if(c.k==="pnl"){const f=num(v);const cls=f>0?"pos":(f<0?"neg":"");return `<td class="num ${cls}">${v===""||v==null?"":esc(v)}</td>`;}
      if(c.n) return `<td class="num">${v===""||v==null?"":esc(v)}</td>`;
      return `<td>${esc(v)}</td>`;
    }).join("");
    const dtr=document.createElement("tr");dtr.className="detail hidden";
    dtr.innerHTML=`<td colspan="${COLS.length}"><div class="io">
        <div><h4>Input (prompt)</h4><pre>${esc(d.input)}</pre></div>
        <div><h4>Output (decision)</h4><pre>${esc(d.output)}</pre></div></div></td>`;
    tr.onclick=()=>dtr.classList.toggle("hidden");
    tb.appendChild(tr);tb.appendChild(dtr);
  });
  document.getElementById("count").textContent=`${r.length} / ${DATA.length} records`;
}
q.oninput=render; render();
</script>
</body></html>"""


def write_html(records: list[dict], out="data_explorer.html"):
    html = HTML.replace("__DATA__", json.dumps(records))
    Path(out).write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({len(records)} records). Open it in a browser.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db")
    p.add_argument("--jsonl", nargs="+")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--out", default="data_explorer.html")
    a = p.parse_args()
    if a.demo:
        recs = demo_records()
    elif a.db:
        recs = from_db(a.db)
    elif a.jsonl:
        recs = from_jsonl(a.jsonl)
    else:
        p.error("give one of --db / --jsonl / --demo")
    write_html(recs, a.out)


if __name__ == "__main__":
    main()
