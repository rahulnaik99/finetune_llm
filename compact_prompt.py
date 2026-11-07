"""
compact_prompt.py
─────────────────
Optimised prompt builder — same information, 62% fewer tokens.
Drop-in replacement for the prompt-building logic in the simulation notebook.

Swap in by replacing the user prompt construction inside pre_trade_decision().
"""
import json
from datetime import datetime


# ── System prompt: concise contract, no repeated JSON schema ─────────────────
SYSTEM_PRE = """You are a disciplined NSE equity trader. Protect capital first.
Rule: R:R must be >= 1.5. Max risk 2% per trade. HOLD if setup unclear.
Output JSON: action|qty|stop_loss|take_profit|risk_reward|reasoning|confidence|key_risk"""

SYSTEM_POST = """Review the completed trade and extract one concrete lesson.
Output JSON: outcome|exit_reason_analysis|what_went_right|what_went_wrong|lesson_learned|strategy_adjustment"""


def _pct(a, b):
    """Percentage change from b to a."""
    if not b or b == 0:
        return None
    return round((a - b) / b * 100, 2)


def _percentile_pos(close, low, high):
    """Where is close in the 5-day range? 0=at low, 100=at high."""
    r = high - low
    if r <= 0:
        return 50
    return round((close - low) / r * 100, 0)


def _body_ratio(candles):
    """Average |close-open| / (high-low) — tells volatility quality."""
    ratios = []
    for c in candles:
        hl = c["high"] - c["low"]
        if hl > 0:
            ratios.append(abs(c["close"] - c["open"]) / hl)
    return round(sum(ratios) / len(ratios), 2) if ratios else 0


def build_compact_user(symbol, date_str, tech, fund, capital, history, initial_cap=10_000):
    """
    Build a compact user prompt.
    Typically 350-420 tokens vs ~990 for the verbose version.
    """
    rc = tech.get("recent_candles", [])
    close = tech.get("close", 0)
    prev_close = rc[-2]["close"] if len(rc) >= 2 else close
    day_chg = _pct(close, prev_close)

    # 5-day range
    d5_high = max(c["high"] for c in rc) if rc else close
    d5_low  = min(c["low"]  for c in rc) if rc else close
    pct_pos  = _percentile_pos(close, d5_low, d5_high)

    # Volume vs 5-day avg
    avg_vol = sum(c.get("volume", 0) for c in rc) / len(rc) if rc else 1
    last_vol = rc[-1].get("volume", 0) if rc else 0
    vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1

    # ATR proxy (avg of last 5 daily ranges)
    daily_ranges = [c["high"] - c["low"] for c in rc] if rc else [0]
    atr_approx = round(sum(daily_ranges) / len(daily_ranges), 2)

    # Momentum
    d3_chg = _pct(close, rc[-4]["close"]) if len(rc) >= 4 else None
    body_r  = _body_ratio(rc)

    # EMA compact
    e50, e100, e200 = tech.get("ema_50"), tech.get("ema_100"), tech.get("ema_200")
    vwap = tech.get("vwap")
    stack = tech.get("ema_stack", "—")
    c_vs_200 = _pct(close, e200) if e200 else None
    c_vs_vwap= _pct(close, vwap) if vwap else None

    # Zones compact  (just show 2 support + 2 resistance, no JSON)
    def fmt_zones(zones, n=2):
        if not zones:
            return "—"
        parts = []
        for z in (zones[:n] if zones else []):
            if z and z.get("low"):
                d = _pct(z["mid"], close) if close else 0
                parts.append(f"{z['low']:.1f}-{z['high']:.1f}({d:+.1f}%)")
        return "  ".join(parts) or "—"

    sup_str = fmt_zones(tech.get("support_zones"))
    res_str = fmt_zones(tech.get("resistance_zones"))
    ns = tech.get("nearest_support")
    nr = tech.get("nearest_resistance")
    dist_s = _pct(ns["mid"], close) if ns and close else None
    dist_r = _pct(nr["mid"], close) if nr and close else None

    # Swings (just the last 2 prices + date, no JSON key overhead)
    sh = tech.get("swing_highs", [])[-2:]
    sl = tech.get("swing_lows", [])[-2:]
    sh_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sh) or "—"
    sl_str = "  ".join(f"{s['price']:.1f}({s['date'][5:]})" for s in sl) or "—"

    # Trade history compact table
    hist_rows = []
    for pt in history[-5:]:
        pnl = pt.get("pnl", 0)
        lesson = (pt.get("lesson_learned") or "—")[:60]
        if lesson in ("—", "", "Analysis unavailable (Ollama not running)"):
            lesson = "—"
        hist_rows.append(
            f"{pt['episode']:02d}  {pt['action']:<4}  "
            f"{pt['exit_reason']:<10}  "
            f"{'+' if pnl>=0 else ''}{pnl:6.0f}  "
            f"{pt['capital_after']:7.0f}  "
            f"{lesson}"
        )
    hist_block = "EP  ACT   RESULT      P&L    CAP     LESSON\n" + \
                 "\n".join(hist_rows) if hist_rows else "(no history)"

    # Fundamentals 1-line (if available)
    fund_line = ""
    if fund and fund.get("net_income"):
        ni = fund["net_income"]
        rv = fund.get("total_revenue")
        pe = fund.get("trailingPE")
        fund_line = (f"\nFund({fund.get('quarter_key','—')}): "
                     f"PAT ₹{ni/1e7:.0f}cr"
                     f"{f' | Rev ₹{rv/1e7:.0f}cr' if rv else ''}"
                     f"{f' | P/E {pe:.1f}' if pe else ''}")

    death = round(initial_cap * 0.20)

    # ── Detail blocks: levels, VIX, zones (computed first so HEADLINE can use them) ──
    lv = tech.get("levels", {}) or {}
    levels_line = ""
    if lv:
        parts = []
        if "pdh" in lv: parts.append(f"PDH {lv['pdh']}({lv.get('pdh_state','')})")
        if "pdl" in lv: parts.append(f"PDL {lv['pdl']}({lv.get('pdl_state','')})")
        if "two_day_high" in lv:
            b = "broke" if lv.get("two_day_high_broken") else "intact"
            parts.append(f"2DH {lv['two_day_high']}({b})")
        if "week_high" in lv: parts.append(f"WkH {lv['week_high']}({lv.get('week_high_state','')})")
        if "week_low" in lv:  parts.append(f"WkL {lv['week_low']}({lv.get('week_low_state','')})")
        if parts:
            levels_line = "LEVELS: " + " | ".join(parts) + "\n"

    vc = tech.get("vix_ctx", {}) or {}
    vix_line = ""
    if vc:
        vix_line = (f"VIX:{vc.get('vix')} | {vc.get('pctile_1y')}pct(1y) | "
                    f"{vc.get('regime')} | 5d:{vc.get('dir_5d')} | 20dRange:{vc.get('pos_20d')}\n")

    zc = tech.get("zones", {}) or {}
    zones_line = ""
    if zc:
        try:
            from zones import fmt_zone
        except Exception:
            fmt_zone = lambda z: "-"
        s15, b1h = zc.get("tf_15m", {}), zc.get("tf_1h", {})
        conf = zc.get("confluence", {}) or {}
        zparts = [f"15m DZ:{fmt_zone(s15.get('demand'))} SZ:{fmt_zone(s15.get('supply'))}",
                  f"1h  DZ:{fmt_zone(b1h.get('demand'))} SZ:{fmt_zone(b1h.get('supply'))}"]
        cf = [k for k in ("demand", "supply") if conf.get(k)]
        if cf: zparts.append("CONFLUENCE:" + "+".join(cf) + " (15m inside 1h)")
        zones_line = "ZONES:\n  " + "\n  ".join(zparts) + "\n"

    # ── RELATIVE encoding (scale-free features for cross-stock generalization) ──
    # Keep raw prices above (for SL/TP math); add % / ATR-multiple distances so the
    # model learns "2% below EMA200, 1.3 ATR from support" not the literal "119.78".
    import os as _os
    rel_on = _os.environ.get("RELATIVE_ENCODING", "1").strip().lower() in ("1", "true", "yes", "on")
    rel_line = ""
    if rel_on:
        atr = atr_approx or 1e-9
        def _atrmult(level):
            return None if level in (None, 0) else round((close - level) / atr, 2)
        ns_mid = (tech.get("nearest_support") or {}).get("mid")
        nr_mid = (tech.get("nearest_resistance") or {}).get("mid")
        rel = [
            f"vsEMA200:{c_vs_200:+.1f}%" if c_vs_200 is not None else "",
            f"vsEMA50:{_pct(close, e50):+.1f}%" if e50 else "",
            f"vsVWAP:{c_vs_vwap:+.1f}%" if c_vs_vwap is not None else "",
            f"ATR%:{round(atr/close*100,2)}%" if close else "",
            f"toSup:{_atrmult(ns_mid)}ATR" if ns_mid else "",
            f"toRes:{_atrmult(nr_mid)}ATR" if nr_mid else "",
            f"5dPos:{pct_pos:.0f}pct",
        ]
        rel_line = "REL: " + " | ".join(p for p in rel if p) + "\n"

    # ── HEADLINE: one-line verdict the model anchors to, then the detail blocks ──
    bias = ("bullish" if (stack or "").startswith("bullish") and (c_vs_vwap or 0) > 0 else
            "bearish" if (stack or "").startswith("bearish") and (c_vs_vwap or 0) < 0 else "neutral")
    hl_bits = [f"bias:{bias}", f"trend:{tech.get('trend','-')}"]
    if lv.get("pdh_state") in ("broke", "rejected"): hl_bits.append(f"PDH:{lv['pdh_state']}")
    if vc: hl_bits.append(f"VIX:{vc.get('regime')}")
    if (zc.get("confluence") or {}).get("demand"): hl_bits.append("confluent-demand")
    if (zc.get("confluence") or {}).get("supply"): hl_bits.append("confluent-supply")
    headline = "HEADLINE: " + " | ".join(hl_bits) + "\n"

    return (
        f"[{symbol} {date_str}]\n"
        f"{headline}"
        f"C:{close} ({'+' if (day_chg or 0)>=0 else ''}{day_chg}%) | "
        f"5d:{d5_low}-{d5_high} C@{pct_pos:.0f}pct\n"
        f"ATR:{atr_approx} | Vol:{last_vol/1e6:.1f}M({vol_ratio}x avg) | "
        f"3d:{'+' if (d3_chg or 0)>=0 else ''}{d3_chg if d3_chg is not None else 0}% | body:{body_r}\n"
        f"{rel_line}\n"

        f"EMA: 50@{e50} {'<' if (e50 or 0)<(e100 or 0) else '>'} "
        f"100@{e100} {'<' if (e100 or 0)<(e200 or 0) else '>'} "
        f"200@{e200} [{stack.split()[0].upper()}]\n"
        f"VWAP:{vwap} C{'+' if (c_vs_vwap or 0)>=0 else ''}{c_vs_vwap}% | "
        f"C vs EMA200:{'+' if (c_vs_200 or 0)>=0 else ''}{c_vs_200}%\n"
        f"Trend:{tech.get('trend','-')} | "
        f"Struct:{','.join(tech.get('structure',[]) or ['-'])}\n\n"

        f"S:{sup_str}  (nearest {(dist_s if dist_s is not None else 0):+.1f}%)" + ("\n" if sup_str != "-" else " -\n") +
        f"R:{res_str}  (nearest {'+' if (dist_r or 0)>=0 else ''}{dist_r if dist_r is not None else 0}%)" +
        ("\n" if res_str != "-" else " -\n") +
        f"SH:{sh_str} | SL:{sl_str}\n"
        f"{levels_line}"
        f"{vix_line}"
        f"{zones_line}"
        f"{fund_line}\n\n"

        f"HISTORY:\n{hist_block}\n\n"

        f"CAP:{capital:.0f} (death<{death})"
    )

def build_compact_post_user(trade):
    """Compact post-trade prompt."""
    pnl_s = f"+{trade['pnl']:.0f}" if trade["pnl"] >= 0 else f"{trade['pnl']:.0f}"
    return (
        f"[{trade['symbol']} Ep{trade['episode']} {trade['decision_date']}]\n"
        f"{trade['action']} {trade['quantity']}@{trade['entry_price']:.2f} "
        f"SL:{trade['stop_loss']:.2f} TP:{trade['take_profit']:.2f}\n"
        f"EXIT:{trade['exit_price']:.2f} [{trade['exit_reason']}] "
        f"on {trade['exit_date'] or '—'}\n"
        f"P&L:{pnl_s} | {trade['capital_before']:.0f}->{trade['capital_after']:.0f}\n\n"
        f"REASONING: {trade['pre_trade_reasoning']}"
    )


# ── Token counter (rough approximation) ───────────────────────────────────────
def count_tokens(text: str) -> int:
    """GPT-style approximation: ~4 chars per token."""
    return len(text) // 4


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Show before/after comparison with sample data
    tech = {
        "close": 119.78, "ema_50": 118.9, "ema_100": 120.1, "ema_200": 122.5,
        "vwap": 119.2, "ema_stack": "mixed", "above_vwap": True, "above_ema200": False,
        "trend": "downtrend", "structure": ["LH", "LL"],
        "nearest_support": {"low": 116.5, "high": 117.2, "mid": 116.85},
        "nearest_resistance": {"low": 122.0, "high": 123.5, "mid": 122.75},
        "support_zones": [{"low": 116.5, "high": 117.2, "mid": 116.85}],
        "resistance_zones": [{"low": 122.0, "high": 123.5, "mid": 122.75}],
        "swing_highs": [{"date": "2024-10-07", "price": 123.5}],
        "swing_lows":  [{"date": "2024-10-09", "price": 116.8}],
        "recent_candles": [
            {"date": "2024-10-06", "open": 121.2, "high": 123.5, "low": 120.1, "close": 121.8, "volume": 4200000},
            {"date": "2024-10-07", "open": 121.5, "high": 122.8, "low": 119.5, "close": 120.2, "volume": 3800000},
            {"date": "2024-10-08", "open": 120.0, "high": 121.2, "low": 118.5, "close": 119.1, "volume": 5100000},
            {"date": "2024-10-09", "open": 119.0, "high": 120.5, "low": 116.8, "close": 117.5, "volume": 6200000},
            {"date": "2024-10-10", "open": 117.8, "high": 120.2, "low": 117.0, "close": 119.78, "volume": 4800000},
        ],
    }
    history = [
        {"episode": 1, "action": "BUY", "exit_reason": "SL_HIT",
         "pnl": -52.0, "capital_after": 9948, "lesson_learned": "wait for EMA alignment"},
    ]
    fund = {"net_income": 53810000000, "total_revenue": 92000000000,
            "trailingPE": 9.2, "quarter_key": "2024-Q2"}

    compact = build_compact_user("CANBK", "2024-10-10", tech, fund, 9948.0, history)

    print("=" * 60)
    print("COMPACT USER PROMPT:")
    print("=" * 60)
    print(compact)
    print()
    print(f"System tokens (compact):  ~{count_tokens(SYSTEM_PRE)}")
    print(f"User tokens   (compact):  ~{count_tokens(compact)}")
    print(f"Total compact:            ~{count_tokens(SYSTEM_PRE) + count_tokens(compact)}")
    print()
    print(f"Total verbose (previous): ~990")
    print(f"Reduction:                ~{100 - (count_tokens(SYSTEM_PRE) + count_tokens(compact)) * 100 // 990}%")
