"""
tests/test_audit2_fixes.py
──────────────────────────
Regression tests for the second audit (look-ahead in history, symbol scoping,
features_used contract, null reasoning, eval-cache fingerprint, resume history,
env comment stripping).
"""
import json
import sqlite3

import common  # noqa: F401 — path bootstrap


def _trade(sym, dd, action="SELL", entry=100.0, sl=105.0, tp=90.0, pnl=10.0, ep=1):
    return {"episode": ep, "symbol": sym, "decision_date": dd, "action": action,
            "entry_price": entry, "stop_loss": sl, "take_profit": tp, "quantity": 10,
            "exit_price": tp, "exit_reason": "TP_HIT", "pnl": pnl,
            "capital_before": 10000.0, "pre_trade_reasoning": "r",
            "lesson_learned": "l", "technicals_json": "{}", "id": ep}


# ── #1 + #2: history is date-filtered AND symbol-scoped ──────────────────────
def test_history_excludes_future_and_other_symbols():
    import data_pipeline as dp
    cur = _trade("CANBK", "2015-01-12", ep=3)
    prev = [
        _trade("CANBK", "2015-01-09", ep=1),     # past, same symbol → include
        _trade("CANBK", "2015-01-14", ep=2),     # FUTURE → must be excluded
        _trade("SBIN",  "2015-01-10", ep=4),     # other symbol → must be excluded
    ]
    ex = dp.build_pre_trade_example(cur, prev)
    prompt = ex["messages"][1]["content"]
    assert "2015-01-09" in prompt            # legit past trade shown
    assert "2015-01-14" not in prompt        # no look-ahead
    assert "SBIN" not in prompt              # no cross-symbol leak


# ── #4: features_used is in the training example + the schema ────────────────
def test_features_used_in_example_and_schema():
    import data_pipeline as dp
    ex = dp.build_pre_trade_example(_trade("CANBK", "2015-02-01"), [])
    d = json.loads(ex["messages"][-1]["content"])
    assert isinstance(d.get("features_used"), list) and d["features_used"]
    assert "features_used" in dp.PRE_TRADE_SYSTEM


# ── #7: reasoning=null must not crash the quality pass ───────────────────────
def test_reasoning_ok_handles_null(tmp_path):
    import evaluate as ev
    f = tmp_path / "trading_test.jsonl"
    f.write_text(json.dumps({"messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"},
        {"role": "assistant", "content": json.dumps({"action": "BUY"})}]}) + "\n")
    # model returns valid JSON with reasoning: null
    q = ev.evaluate_model_quality(str(f), lambda s, u: '{"action":"BUY","reasoning":null}')
    assert q["n"] == 1                        # did not raise


# ── #6: eval cache is rejected when the test-set fingerprint changes ─────────
def test_eval_cache_fingerprint(tmp_path, monkeypatch):
    from modules.evaluation import service as es
    monkeypatch.setattr(es, "METRICS_DIR", tmp_path)
    (tmp_path / "eval_round_01.json").write_text(json.dumps({"round": 1, "_fingerprint": "AAAA"}))
    assert es.load_cached(1, fingerprint="AAAA") is not None     # match → reuse
    assert es.load_cached(1, fingerprint="BBBB") is None         # stale → ignore
    assert es.load_cached(1) is not None                         # no fp → lenient


# ── #5: history_from_db returns chronological same-symbol rows ───────────────
def test_history_from_db(tmp_path):
    from modules.datagen import service as svc
    import simulation  # noqa  (ensures schema helpers available)
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript(svc._TRADE_LOG_DDL)
    con.executemany("INSERT INTO trade_log (episode,symbol,decision_date,action,entry_price,pnl) "
                    "VALUES (?,?,?,?,?,?)",
                    [(2, "CANBK", "2015-01-12", "SELL", 49.0, 5.0),
                     (1, "CANBK", "2015-01-09", "SELL", 50.0, -3.0),
                     (1, "SBIN",  "2015-01-10", "BUY", 200.0, 7.0)])
    con.commit(); con.close()
    hist = svc.history_from_db("CANBK", str(db))
    assert [h["decision_date"] for h in hist] == ["2015-01-09", "2015-01-12"]  # chrono, CANBK only


# ── LOW: .env values containing '#' survive ──────────────────────────────────
def test_env_hash_in_value_preserved(tmp_path):
    import config, os
    p = tmp_path / ".env"
    p.write_text("TOKEN=abc#def\nURL=http://x/y#frag  # real comment\n")
    for k in ("TOKEN", "URL"):
        os.environ.pop(k, None)
    config.load_env(str(p))
    assert config.env_str("TOKEN", "") == "abc#def"
    assert config.env_str("URL", "") == "http://x/y#frag"
