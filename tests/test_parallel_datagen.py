"""
tests/test_parallel_datagen.py
──────────────────────────────
Tests for parallel (multi-symbol) data generation. The CPU-heavy simulation is
not executed here (it needs market data + time); instead we validate the parts
that are new and bug-prone: the shard→main merge (idempotent, per-symbol
replace) and the API contract.
"""
import sqlite3

import common  # noqa: F401 — path bootstrap
from fastapi.testclient import TestClient

from app.main import app
from modules.datagen import service as svc

client = TestClient(app)


def _make_shard(path, symbol, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(svc._TRADE_LOG_DDL)
    con.executemany(
        "INSERT INTO trade_log (episode, symbol, decision_date, action, pnl) "
        "VALUES (?,?,?,?,?)",
        [(i, symbol, "2024-01-%02d" % (i + 1), "SELL", 10.0 * i) for i in range(rows)],
    )
    con.commit(); con.close()


def test_merge_is_idempotent_and_per_symbol(tmp_path):
    main = tmp_path / "trading_agent.db"
    # two symbols, each with its own shard at the expected path
    for sym, n in [("CANBK", 3), ("SBIN", 2)]:
        _make_shard(svc._shard_db_path(sym, str(main)), sym, n)

    assert svc._merge_shard_into_main("CANBK", str(main)) == 3
    assert svc._merge_shard_into_main("SBIN", str(main)) == 2
    # merging CANBK again must NOT duplicate rows (idempotent replace)
    assert svc._merge_shard_into_main("CANBK", str(main)) == 3

    con = sqlite3.connect(main)
    total = con.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]
    canbk = con.execute("SELECT COUNT(*) FROM trade_log WHERE symbol='CANBK'").fetchone()[0]
    con.close()
    assert total == 5 and canbk == 3           # both symbols preserved, no dupes


def test_shard_path_is_per_symbol(tmp_path):
    a = svc._shard_db_path("CANBK", str(tmp_path / "db.db"))
    b = svc._shard_db_path("SBIN", str(tmp_path / "db.db"))
    assert a != b and a.parent.name == "shards"


def test_parallel_endpoint_validates_input():
    # symbols is required → 422 when missing
    r = client.post("/api/datagen/run_parallel", json={"episodes": 10})
    assert r.status_code == 422


def test_parallel_endpoint_in_openapi():
    spec = client.get("/openapi.json").json()
    assert "/api/datagen/run_parallel" in spec["paths"]
