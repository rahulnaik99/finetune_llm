"""
tests/test_build.py
───────────────────
Build & smoke tests for the whole llm_finetune package.

Covers:
  - every .py file parses (build check)
  - heavy deps (yfinance/requests/mlx) are stubbed so modules import on any box
  - pure-function units: trading_metrics, balance_and_dedup, zones.detect_zones,
    config.load_env, compact_prompt rendering, XBRL parsing

Run:
    pytest tests/test_build.py -q
    #  or without pytest:
    python tests/test_build.py
"""
import ast, json, sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── stub heavy / optional deps so imports never fail in CI ──────────────────
def _stub_heavy_modules():
    if "yfinance" not in sys.modules:
        yf = types.ModuleType("yfinance")
        yf.download = lambda *a, **k: None
        yf.Ticker = lambda *a, **k: types.SimpleNamespace(quarterly_financials=None, info={})
        sys.modules["yfinance"] = yf
    if "requests" not in sys.modules:
        rq = types.ModuleType("requests")
        rq.post = lambda *a, **k: None
        sys.modules["requests"] = rq


_stub_heavy_modules()


# ── 1. BUILD: every python file parses ──────────────────────────────────────
def test_all_files_compile():
    py_files = [p for p in ROOT.rglob("*.py") if "__pycache__" not in str(p)]
    assert py_files, "no python files found"
    for p in py_files:
        src = p.read_text(encoding="utf-8")
        ast.parse(src)   # raises SyntaxError on a bad build


# ── 2. core modules import ──────────────────────────────────────────────────
def test_core_imports():
    import config, zones, evaluate, compact_prompt, fundamentals_xbrl  # noqa
    import data_pipeline, simulation                                   # noqa


# ── 3. trading metrics math ─────────────────────────────────────────────────
def test_trading_metrics():
    import evaluate as ev
    m = ev.trading_metrics([52, -80, 237, 313, -26, -149, 52, -83])
    assert m["n"] == 8
    assert 0 <= m["win_rate"] <= 100
    assert m["profit_factor"] > 0
    assert "max_drawdown" in m and m["max_drawdown"] <= 0
    assert ev.trading_metrics([]) == {"n": 0}


# ── 4. balancer + de-dup ─────────────────────────────────────────────────────
def test_balance_and_dedup():
    import data_pipeline as dp
    def ex(a, sl, tp, rr, c):
        return {"messages": [{"role": "assistant",
                "content": json.dumps({"action": a, "stop_loss": sl, "take_profit": tp,
                                       "risk_reward": rr, "confidence": c})}]}
    examples = ([ex("BUY", 116, 123, 1.8, 0.7)] * 6 +
                [ex("BUY", 110, 120, 2.0, 0.6), ex("BUY", 100, 112, 1.9, 0.65),
                 ex("SELL", 125, 118, 1.7, 0.6), ex("SELL", 130, 120, 2.1, 0.7)] +
                [ex("HOLD", 0, 0, 0, 0.4)] * 3)
    out = dp.balance_and_dedup(examples, dedup=True, balance=True)
    acts = [dp._assistant_json(e).get("action") for e in out]
    assert len(out) < len(examples)             # dupes removed
    assert abs(acts.count("BUY") - acts.count("SELL")) <= acts.count("SELL")  # balanced-ish


# ── 5. zones detector ────────────────────────────────────────────────────────
def test_zones():
    import pandas as pd, numpy as np, zones
    # base (tight) then an impulse up → should yield a demand zone
    o = [100, 100, 100, 100, 106, 107]
    h = [100.3, 100.2, 100.2, 100.3, 106.5, 107.5]
    l = [99.8, 99.8, 99.9, 99.8, 100.2, 106.0]
    c = [100.1, 100.0, 100.1, 100.1, 106.2, 107.2]
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": [0]*6})
    z = zones.detect_zones(df, base_max=3, impulse_mult=3.0, scan=20)
    assert any(zz["kind"] == "demand" for zz in z)


# ── 6. config .env loader ────────────────────────────────────────────────────
def test_config_env(tmp_path=None):
    import config, os
    p = ROOT / "_test_env_tmp"
    p.write_text("FOO_BAR=hello\nNUM=7\nFLAG=1\n")
    os.environ.pop("FOO_BAR", None); os.environ.pop("NUM", None); os.environ.pop("FLAG", None)
    config.load_env(str(p))
    assert config.env_str("FOO_BAR", "x") == "hello"
    assert config.env_int("NUM", 0) == 7
    assert config.env_bool("FLAG") is True
    p.unlink()


# ── 7. compact prompt renders with REL + HEADLINE ───────────────────────────
def test_prompt_renders():
    from compact_prompt import build_compact_user
    tech = {"close": 119.78, "ema_50": 118.9, "ema_100": 120.1, "ema_200": 122.5, "vwap": 119.2,
            "ema_stack": "bearish (50<100<200)", "above_vwap": True, "above_ema200": False,
            "trend": "downtrend", "structure": ["LH", "LL"],
            "nearest_support": {"low": 116.5, "high": 117.2, "mid": 116.85},
            "nearest_resistance": {"low": 122.0, "high": 123.5, "mid": 122.75},
            "support_zones": [], "resistance_zones": [], "swing_highs": [], "swing_lows": [],
            "recent_candles": [{"date": "2024-01-10", "open": 118, "high": 120, "low": 117,
                                "close": 119.78, "volume": 4_800_000}]}
    p = build_compact_user("CANBK", "2024-01-10", tech, {}, 10000, [])
    assert "HEADLINE:" in p and "REL:" in p and "CANBK" in p


# ── 8. XBRL fundamentals parse (if the sample file is present) ──────────────
def test_xbrl_parse():
    import fundamentals_xbrl as fx
    sample = ROOT / "data" / "fundamentals" / "CANBK" / "CANBK_2026Q2.xml"
    if not sample.exists():
        return  # sample not shipped in this build — skip
    f = fx.parse_xbrl_file(sample)
    assert f.get("symbol")                 # some symbol parsed
    assert f.get("gnpa_pct") is not None


# ── allow running without pytest ─────────────────────────────────────────────
if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
