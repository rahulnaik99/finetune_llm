"""
fundamentals_xbrl.py
────────────────────
Parse SEBI "Integrated Filing - Banking" XBRL quarterly results and build a
single, LLM-ready JSON per symbol with ALL derived metrics PRE-COMPUTED
(QoQ, YoY, TTM, CD-ratio, NII, asset-quality deltas). The model just reads
the numbers — no arithmetic required at inference time.

Handles a folder containing many filings:
  - de-duplicates Original vs Revision (keeps the latest/most-complete)
  - separates Standalone vs Consolidated
  - sorts quarters and computes period-over-period changes
  - gates by result date (no look-ahead)

CLI:
    # build data/fundamentals/CANBK/fundamentals.json (consolidated)
    python fundamentals_xbrl.py --build CANBK --dir data/fundamentals --nature Consolidated

    # inspect one file
    python fundamentals_xbrl.py one_file.xml
"""
from __future__ import annotations
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


# ════════════════════════════════════════════════════════════════════
#  Low-level parse
# ════════════════════════════════════════════════════════════════════
def _local(tag: str) -> str:
    return tag.split("}")[-1]


def _f(txt):
    try:
        return float(txt)
    except (TypeError, ValueError):
        return None


def parse_xbrl_file(path: str | Path) -> dict:
    """Extract the key quarterly + balance-sheet facts from one XBRL filing."""
    root = ET.parse(path).getroot()

    contexts = {}
    for ctx in root.findall("{*}context"):
        cid = ctx.get("id")
        period = ctx.find("{*}period")
        start = period.findtext("{*}startDate") if period is not None else None
        end = period.findtext("{*}endDate") if period is not None else None
        instant = period.findtext("{*}instant") if period is not None else None
        scenario = ctx.find("{*}scenario")
        has_dim = scenario is not None and len(list(scenario)) > 0
        contexts[cid] = {"start": start, "end": end, "instant": instant, "has_dim": has_dim}

    facts: dict[str, dict] = {}
    for el in root.iter():
        cref = el.get("contextRef")
        if not cref:
            continue
        facts.setdefault(_local(el.tag), {})[cref] = (el.text or "").strip()

    def _days(c):
        try:
            return (datetime.fromisoformat(c["end"]) - datetime.fromisoformat(c["start"])).days
        except Exception:
            return None

    qtr_ctx = None
    for cid, c in contexts.items():
        if c["start"] and c["end"] and not c["has_dim"]:
            d = _days(c)
            if d is not None and 80 <= d <= 100:
                qtr_ctx = cid
                break
    inst_ctx, latest = None, ""
    for cid, c in contexts.items():
        if c["instant"] and not c["has_dim"] and c["instant"] > latest:
            latest, inst_ctx = c["instant"], cid

    def q(name): return facts.get(name, {}).get(qtr_ctx)
    def i(name): return facts.get(name, {}).get(inst_ctx)
    def any_(name):
        d = facts.get(name, {})
        return next(iter(d.values()), None) if d else None

    pat = _f(q("ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates")) \
          or _f(q("ProfitLossForThePeriod"))

    return {
        "symbol": any_("Symbol"),
        "name": any_("NameOfBank"),
        "isin": any_("ISIN"),
        "quarter_end": contexts.get(qtr_ctx, {}).get("end"),
        "reporting_quarter": any_("ReportingQuarter"),
        "result_date": any_("DateOfBoardMeetingWhenFinancialResultsWereApproved"),
        "audited": any_("WhetherResultsAreAuditedOrUnaudited"),
        "nature": any_("NatureOfReportStandaloneConsolidated"),
        # income statement (quarter)
        "interest_earned": _f(q("InterestEarned")),
        "interest_expended": _f(q("InterestExpended")),
        "other_income": _f(q("OtherIncome")),
        "total_income": _f(q("Income")),
        "operating_expenses": _f(q("OperatingExpenses")),
        "employees_cost": _f(q("EmployeesCost")),
        "operating_profit": _f(q("OperatingProfitBeforeProvisionAndContingencies")),
        "provisions": _f(q("ProvisionsOtherThanTaxAndContingencies")),
        "tax": _f(q("TaxExpense")),
        "pat": pat,
        "eps": _f(q("BasicEarningsPerShareAfterExtraordinaryItems")),
        # asset quality / ratios
        "gnpa": _f(q("GrossNonPerformingAssets")),
        "nnpa": _f(q("NonPerformingAssets")),
        "gnpa_pct": _f(q("PercentageOfGrossNpa")),
        "nnpa_pct": _f(q("PercentageOfNpa")),
        "cet1_ratio": _f(q("CET1Ratio")),
        "roa": _f(q("ReturnOnAssets")),
        # balance sheet (instant)
        "deposits": _f(i("Deposits")),
        "advances": _f(i("Advances")),
        "total_assets": _f(i("Assets")),
        "capital": _f(i("Capital")),
        "reserves": _f(i("ReservesAndSurplus")),
    }


# ════════════════════════════════════════════════════════════════════
#  Derived-metric helpers
# ════════════════════════════════════════════════════════════════════
def _pct(cur, prev):
    if cur is None or prev in (None, 0):
        return None
    return round((cur / prev - 1) * 100, 2)


def _bps(cur, prev):
    """Change in basis points for ratios already expressed as fractions (0.0133)."""
    if cur is None or prev is None:
        return None
    return round((cur - prev) * 10000, 1)


def _cr(v):
    return round(v / 1e7, 1) if isinstance(v, (int, float)) else None   # INR -> crore


# ════════════════════════════════════════════════════════════════════
#  Batch build → LLM-ready JSON
# ════════════════════════════════════════════════════════════════════
def _field_count(d: dict) -> int:
    return sum(1 for v in d.values() if v not in (None, "", 0))


def load_and_dedup(symbol: str, xbrl_dir: str, nature: str) -> list[dict]:
    """
    Parse every XBRL for `symbol`, keep only the requested nature
    (Consolidated/Standalone), and de-duplicate per quarter_end by choosing
    the most-complete / latest filing (handles Original vs Revision).
    """
    base = Path(xbrl_dir)
    sub = base / symbol
    # rglob = recurse into subfolders, so data/fundamentals/CANBK/2014/*.xml works too.
    files = (list(sub.rglob("*.xml")) + list(sub.rglob("*.xbrl"))) if sub.is_dir() else \
            [p for p in base.rglob("*") if symbol.upper() in p.name.upper()
             and p.suffix.lower() in (".xml", ".xbrl")]

    by_quarter: dict[str, dict] = {}
    for p in files:
        try:
            f = parse_xbrl_file(p)
        except Exception as e:
            print(f"  [skip {p.name}: {e}]")
            continue
        if not f.get("quarter_end"):
            continue
        if nature and f.get("nature") and f["nature"].lower() != nature.lower():
            continue
        f["_file"] = p.name
        f["_mtime"] = p.stat().st_mtime
        key = f["quarter_end"]
        cur = by_quarter.get(key)
        # keep the better one: more populated, then newer file (revision wins)
        if cur is None or (_field_count(f), f["_mtime"]) > (_field_count(cur), cur["_mtime"]):
            by_quarter[key] = f

    return [by_quarter[k] for k in sorted(by_quarter)]


def build_fundamentals_json(symbol: str, xbrl_dir: str = "data/fundamentals",
                            nature: str = "Consolidated",
                            out_path: str | None = None) -> dict:
    """
    Build the LLM-ready JSON with QoQ/YoY/TTM and ratios pre-computed.
    Writes to {xbrl_dir}/{symbol}/fundamentals.json by default.
    """
    rows = load_and_dedup(symbol, xbrl_dir, nature)
    if not rows:
        raise FileNotFoundError(f"No {nature} XBRL filings for {symbol} in {xbrl_dir}")

    # index helpers for QoQ (previous row) and YoY (~365d earlier)
    ends = [datetime.fromisoformat(r["quarter_end"]) for r in rows]

    def yoy_partner(idx):
        target = ends[idx].replace(year=ends[idx].year - 1)
        best, bestdiff = None, 40  # within 40 days
        for j, e in enumerate(ends):
            diff = abs((e - target).days)
            if diff < bestdiff:
                best, bestdiff = j, diff
        return best

    quarters = []
    for idx, r in enumerate(rows):
        prev = rows[idx - 1] if idx > 0 else None
        yj = yoy_partner(idx)
        yo = rows[yj] if yj is not None and yj != idx else None

        nii = None
        if r.get("interest_earned") is not None and r.get("interest_expended") is not None:
            nii = r["interest_earned"] - r["interest_expended"]
        nii_prev = None
        if prev and prev.get("interest_earned") is not None and prev.get("interest_expended") is not None:
            nii_prev = prev["interest_earned"] - prev["interest_expended"]
        nii_yo = None
        if yo and yo.get("interest_earned") is not None and yo.get("interest_expended") is not None:
            nii_yo = yo["interest_earned"] - yo["interest_expended"]

        # TTM EPS = sum of last 4 quarters' EPS (this + 3 preceding)
        eps_vals = [x.get("eps") for x in rows[max(0, idx - 3):idx + 1] if x.get("eps") is not None]
        eps_ttm = round(sum(eps_vals), 2) if len(eps_vals) == 4 else None

        cd_ratio = None
        if r.get("advances") and r.get("deposits"):
            cd_ratio = round(r["advances"] / r["deposits"] * 100, 1)

        # provision coverage proxy (how much provisioning vs gross NPA stock)
        pcr_hint = None
        if r.get("gnpa") and r.get("nnpa") is not None and r["gnpa"]:
            pcr_hint = round((1 - r["nnpa"] / r["gnpa"]) * 100, 1)

        qd = {
            "quarter_end": r["quarter_end"],
            "reporting_quarter": r.get("reporting_quarter"),
            "result_date": r.get("result_date"),
            "nature": r.get("nature"),
            "audited": r.get("audited"),

            "pat_cr": _cr(r.get("pat")),
            "pat_qoq_pct": _pct(r.get("pat"), prev.get("pat") if prev else None),
            "pat_yoy_pct": _pct(r.get("pat"), yo.get("pat") if yo else None),

            "net_interest_income_cr": _cr(nii),
            "nii_qoq_pct": _pct(nii, nii_prev),
            "nii_yoy_pct": _pct(nii, nii_yo),

            "total_income_cr": _cr(r.get("total_income")),
            "income_yoy_pct": _pct(r.get("total_income"), yo.get("total_income") if yo else None),

            "operating_profit_cr": _cr(r.get("operating_profit")),
            "opprofit_yoy_pct": _pct(r.get("operating_profit"), yo.get("operating_profit") if yo else None),

            "provisions_cr": _cr(r.get("provisions")),
            "eps": r.get("eps"),
            "eps_ttm": eps_ttm,

            "gnpa_pct": round(r["gnpa_pct"] * 100, 2) if r.get("gnpa_pct") is not None else None,
            "gnpa_qoq_bps": _bps(r.get("gnpa_pct"), prev.get("gnpa_pct") if prev else None),
            "gnpa_yoy_bps": _bps(r.get("gnpa_pct"), yo.get("gnpa_pct") if yo else None),
            "nnpa_pct": round(r["nnpa_pct"] * 100, 2) if r.get("nnpa_pct") is not None else None,
            "provision_coverage_pct": pcr_hint,

            "cet1_pct": round(r["cet1_ratio"] * 100, 2) if r.get("cet1_ratio") is not None else None,
            "roa_pct": round(r["roa"] * 100, 2) if r.get("roa") is not None else None,

            "deposits_cr": _cr(r.get("deposits")),
            "advances_cr": _cr(r.get("advances")),
            "cd_ratio_pct": cd_ratio,
            "deposit_yoy_pct": _pct(r.get("deposits"), yo.get("deposits") if yo else None),
            "advance_yoy_pct": _pct(r.get("advances"), yo.get("advances") if yo else None),

            "source_file": r.get("_file"),
        }
        qd["summary"] = _summary_line(qd)
        quarters.append(qd)

    out = {
        "symbol": rows[-1].get("symbol") or symbol,
        "name": rows[-1].get("name"),
        "nature": nature,
        "n_filings": len(rows),
        "latest_quarter": rows[-1]["quarter_end"],
        "quarters": quarters,
        "llm_ready_latest": quarters[-1]["summary"],
    }

    out_path = out_path or str(Path(xbrl_dir) / symbol / "fundamentals.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"  Wrote {out_path}  ({len(quarters)} quarters, {nature})")
    return out


def _summary_line(q: dict) -> str:
    """Human/LLM-readable one-liner with the pre-computed deltas."""
    def g(k, suf="", sign=False):
        v = q.get(k)
        if v is None:
            return "n/a"
        s = f"{v:+}" if sign else f"{v}"
        return f"{s}{suf}"
    return (
        f"{q['reporting_quarter'] or ''} (end {q['quarter_end']}): "
        f"PAT Rs.{g('pat_cr')}cr (YoY {g('pat_yoy_pct','%',True)}, QoQ {g('pat_qoq_pct','%',True)}); "
        f"NII Rs.{g('net_interest_income_cr')}cr (YoY {g('nii_yoy_pct','%',True)}); "
        f"EPS Rs.{g('eps')} (TTM Rs.{g('eps_ttm')}); "
        f"GNPA {g('gnpa_pct','%')} (YoY {g('gnpa_yoy_bps','bps',True)}, QoQ {g('gnpa_qoq_bps','bps',True)}), "
        f"NNPA {g('nnpa_pct','%')}, PCR {g('provision_coverage_pct','%')}; "
        f"CET1 {g('cet1_pct','%')}, ROA {g('roa_pct','%')}; "
        f"Deposits Rs.{g('deposits_cr')}cr (YoY {g('deposit_yoy_pct','%',True)}), "
        f"Advances Rs.{g('advances_cr')}cr (YoY {g('advance_yoy_pct','%',True)}), "
        f"CD-ratio {g('cd_ratio_pct','%')}."
    )


# ════════════════════════════════════════════════════════════════════
#  Lookups used by the simulation (gated by result date = no look-ahead)
# ════════════════════════════════════════════════════════════════════
def get_fundamentals_xbrl(symbol: str, as_of: str, xbrl_dir: str = "data/fundamentals",
                          nature: str = "Consolidated") -> dict:
    """
    Return the most recent PRE-COMPUTED quarter whose result date <= as_of.
    Reads the cached fundamentals.json if present, else builds it.
    """
    jpath = Path(xbrl_dir) / symbol / "fundamentals.json"
    data = json.loads(jpath.read_text()) if jpath.exists() else \
           build_fundamentals_json(symbol, xbrl_dir, nature)
    usable = [q for q in data["quarters"]
              if (q.get("result_date") or q["quarter_end"]) <= as_of]
    if not usable:
        return {}
    q = dict(usable[-1])
    q["net_income"] = (q.get("pat_cr") or 0) * 1e7      # back-compat for prompt builder
    q["quarter_key"] = q["quarter_end"]
    return q


def compact_fundamentals(f: dict) -> str:
    return f.get("summary", "Fund: n/a") if f else "Fund: n/a"


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--build":
        symbol = sys.argv[2]
        xdir = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "data/fundamentals"
        nat = sys.argv[sys.argv.index("--nature") + 1] if "--nature" in sys.argv else "Consolidated"
        data = build_fundamentals_json(symbol, xdir, nat)
        print("\nLATEST:", data["llm_ready_latest"])
    else:
        print(json.dumps(parse_xbrl_file(sys.argv[1]), indent=2))
