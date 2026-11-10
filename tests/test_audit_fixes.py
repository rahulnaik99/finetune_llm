"""
tests/test_audit_fixes.py
─────────────────────────
Regression tests locking the Day-5 audit fixes so they can't silently come back.
"""
import json
import sqlite3

import common  # noqa: F401 — path bootstrap


# ── 1. SELL risk:reward is computed (was hardcoded 0.0 → every SELL dropped) ──
def test_sell_risk_reward_and_survives_filter():
    import data_pipeline as dp
    trade = {
        "episode": 1, "symbol": "CANBK", "decision_date": "2024-02-01",
        "action": "SELL", "entry_price": 100.0, "stop_loss": 105.0,
        "take_profit": 90.0, "quantity": 10, "pnl": 50.0, "exit_reason": "TP_HIT",
        "pre_trade_reasoning": "bearish stack below VWAP", "capital_before": 10000.0,
        "technicals_json": "{}",
    }
    ex = dp.build_pre_trade_example(trade, [])
    d = json.loads(ex["messages"][-1]["content"])
    assert d["action"] == "SELL"
    assert d["risk_reward"] == 2.0                      # (100-90)/(105-100)
    assert dp.quality_filter(ex, min_confidence=0.5, min_rr=1.3) is True


# ── 2. post-trade examples no longer collapse to one dedup key ───────────────
def test_post_trade_examples_not_collapsed():
    import data_pipeline as dp

    def post(lesson):
        return {"messages": [{"role": "assistant",
                "content": json.dumps({"outcome": "loss", "lesson_learned": lesson})}],
                "_meta": {"type": "post_trade"}}
    a, b, c = post("cut losses faster"), post("respect the zone"), post("cut losses faster")
    out = dp.balance_and_dedup([a, b, c], dedup=True, balance=False)
    # two DISTINCT lessons survive; the exact duplicate of 'a' is removed
    assert len(out) == 2
    keys = {dp._dedup_key(x) for x in [a, b]}
    assert len(keys) == 2                               # distinct lessons → distinct keys


# ── 3. profit_factor is JSON-safe (no float('inf') → no API 500) ─────────────
def test_profit_factor_json_safe():
    import evaluate as ev
    m = ev.trading_metrics([10, 20, 30])               # all wins → undefined PF
    assert m["profit_factor"] is None
    json.dumps(m, allow_nan=False)                      # must not raise


# ── 4. state filenames are filesystem-safe (no ':' from per-symbol stages) ───
def test_state_filename_is_fs_safe(tmp_path):
    from common.state import StateStore
    st = StateStore("datagen:CANBK:w0", state_dir=tmp_path)
    assert ":" not in st.path.name
    st.start(total=1); assert st.path.exists()


# ── 5. shard merge is NON-destructive (keeps earlier rows) ───────────────────
def test_merge_is_non_destructive(tmp_path):
    from modules.datagen import service as svc
    main = tmp_path / "trading_agent.db"
    # pre-existing CANBK row from an earlier (sequential) run
    con = sqlite3.connect(main); con.executescript(svc._TRADE_LOG_DDL)
    con.execute("INSERT INTO trade_log (episode,symbol,decision_date,action,entry_price,pnl) "
                "VALUES (1,'CANBK','2020-01-01','BUY',50.0,99.0)")
    con.commit(); con.close()
    # a shard with DIFFERENT rows
    shard = svc._shard_db_path("CANBK", str(main))
    shard.parent.mkdir(parents=True, exist_ok=True)
    c2 = sqlite3.connect(shard); c2.executescript(svc._TRADE_LOG_DDL)
    c2.executemany("INSERT INTO trade_log (episode,symbol,decision_date,action,entry_price,pnl) "
                   "VALUES (?,?,?,?,?,?)",
                   [(i, "CANBK", "2024-02-%02d" % (i+1), "SELL", 100.0+i, 10.0*i) for i in range(3)])
    c2.commit(); c2.close()

    svc._merge_shard_into_main("CANBK", str(main))
    con = sqlite3.connect(main)
    total = con.execute("SELECT COUNT(*) FROM trade_log WHERE symbol='CANBK'").fetchone()[0]
    kept = con.execute("SELECT COUNT(*) FROM trade_log WHERE decision_date='2020-01-01'").fetchone()[0]
    con.close()
    assert kept == 1 and total == 4                     # old row preserved + 3 new


# ── 6. model-quality keeps post-trade rows out of schema/action scores ───────
def test_model_quality_separates_row_types(tmp_path):
    import evaluate as ev
    f = tmp_path / "trading_test.jsonl"
    pre = {"messages": [{"role": "system", "content": "s"},
                        {"role": "user", "content": "u1"},
                        {"role": "assistant", "content": json.dumps({
                            "action": "BUY", "stop_loss": 1, "take_profit": 2,
                            "risk_reward": 2, "reasoning": "x", "confidence": 0.7})}]}
    post = {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "u2"},
                         {"role": "assistant", "content": json.dumps({
                             "outcome": "loss", "lesson_learned": "cut"})}]}
    f.write_text(json.dumps(pre) + "\n" + json.dumps(post) + "\n")
    q = ev.evaluate_model_quality(str(f), lambda s, u: pre["messages"][-1]["content"]
                                  if u == "u1" else post["messages"][-1]["content"])
    assert q["n"] == 2 and q["n_decision"] == 1
    assert q["action_match"] == 100.0                   # the single decision row matched
