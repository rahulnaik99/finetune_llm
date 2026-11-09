"""
zones.py
────────
Demand / Supply zone detection using the classic "base + impulse" pattern:
a tight consolidation (base) immediately followed by a strong impulse move.

  - Demand zone  = base, then a strong RALLY away  → buyers' footprint (support).
  - Supply zone  = base, then a strong DROP away    → sellers' footprint (resistance).

Settings (per your choice):
  - impulse_mult = 3.0   → impulse leg must be >= 3x the base range (fewer, cleaner zones)
  - allow_retest = True  → a zone stays valid through ONE retest; invalidated after the
                           2nd touch or once price closes through it (mitigated)

Pure functions — no I/O. The simulation calls these on 15-min and 1-hour candles.
"""
from __future__ import annotations
import pandas as pd


def detect_zones(df: pd.DataFrame, base_max: int = 3, impulse_mult: float = 3.0,
                 scan: int = 120, allow_retest: bool = True) -> list[dict]:
    """
    Scan the last `scan` candles of `df` (OHLC) for demand/supply zones.
    Returns a list of dicts: {kind, low, high, formed, touches, fresh, valid}.
    """
    if df is None or len(df) < base_max + 3:
        return []
    d = df.tail(scan).reset_index(drop=True)
    highs, lows = d["high"].values, d["low"].values
    opens, closes = d["open"].values, d["close"].values
    n = len(d)
    zones = []

    # i = index of the impulse candle; base is the few candles right before it
    for i in range(base_max, n - 1):
        impulse_rng = highs[i] - lows[i]
        if impulse_rng <= 0:
            continue
        for nb in range(1, base_max + 1):
            base = slice(i - nb, i)
            base_hi = highs[base].max()
            base_lo = lows[base].min()
            base_rng = base_hi - base_lo
            if base_rng <= 0:
                continue
            if impulse_rng < impulse_mult * base_rng:
                continue
            bullish = closes[i] > opens[i] and closes[i] > base_hi
            bearish = closes[i] < opens[i] and closes[i] < base_lo
            if not (bullish or bearish):
                continue
            kind = "demand" if bullish else "supply"
            lo, hi = float(base_lo), float(base_hi)

            # mitigation / touches by later candles
            touches, mitigated = 0, False
            for j in range(i + 1, n):
                overlap = lows[j] <= hi and highs[j] >= lo
                if overlap:
                    touches += 1
                if kind == "demand" and closes[j] < lo:
                    mitigated = True; break
                if kind == "supply" and closes[j] > hi:
                    mitigated = True; break
            valid = (not mitigated) and (touches <= (1 if allow_retest else 0))
            zones.append({"kind": kind, "low": round(lo, 2), "high": round(hi, 2),
                          "formed": i, "touches": touches,
                          "fresh": touches == 0, "valid": valid})
            break  # one zone per impulse candle (take the first base that qualifies)
    return zones


def nearest_zones(zones: list[dict], price: float) -> dict:
    """Nearest VALID demand (below price) and supply (above price), with distance %."""
    demands = [z for z in zones if z["kind"] == "demand" and z["valid"] and z["high"] <= price * 1.002]
    supplies = [z for z in zones if z["kind"] == "supply" and z["valid"] and z["low"] >= price * 0.998]
    nd = max(demands, key=lambda z: z["high"]) if demands else None   # closest below
    ns = min(supplies, key=lambda z: z["low"]) if supplies else None  # closest above
    out = {}
    if nd:
        nd = dict(nd); nd["dist_pct"] = round((nd["high"] / price - 1) * 100, 2)
        out["demand"] = nd
    if ns:
        ns = dict(ns); ns["dist_pct"] = round((ns["low"] / price - 1) * 100, 2)
        out["supply"] = ns
    return out


def _overlap(a: dict | None, b: dict | None) -> bool:
    if not a or not b:
        return False
    return a["low"] <= b["high"] and b["low"] <= a["high"]


def confluence(near_short: dict, near_big: dict) -> dict:
    """True where a short-TF zone sits inside/overlaps the big-TF zone of the same kind."""
    return {
        "demand": _overlap(near_short.get("demand"), near_big.get("demand")),
        "supply": _overlap(near_short.get("supply"), near_big.get("supply")),
    }


def fmt_zone(z: dict | None) -> str:
    if not z:
        return "—"
    tag = "fresh" if z.get("fresh") else f"tested({z.get('touches')})"
    return f"{z['low']}-{z['high']} {tag}({z.get('dist_pct', 0):+.1f}%)"
