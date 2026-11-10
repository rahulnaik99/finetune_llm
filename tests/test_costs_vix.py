"""
tests/test_costs_vix.py
───────────────────────
Fees/taxes cost model + VIX-based position sizing. These import the (heavy)
simulation module, which the suite already loads elsewhere, so the import is
cached and cheap here.
"""
import importlib
import os

import common  # noqa: F401 — path bootstrap


def _sim():
    return importlib.import_module("simulation")


def test_costs_positive_for_trade_and_zero_for_hold():
    sim = _sim()
    sim.INCLUDE_COSTS = True
    c = sim.trade_costs(100.0, 102.0, 50, "BUY")
    assert c > 0                       # round-trip fees+taxes on ~10k turnover
    assert sim.trade_costs(100.0, 102.0, 50, "HOLD") == 0.0
    assert sim.trade_costs(100.0, 102.0, 0, "BUY") == 0.0


def test_costs_toggle_off():
    sim = _sim()
    saved = sim.INCLUDE_COSTS
    sim.INCLUDE_COSTS = False
    try:
        assert sim.trade_costs(100.0, 102.0, 50, "BUY") == 0.0
    finally:
        sim.INCLUDE_COSTS = saved


def test_vix_sizing_low_vs_high():
    sim = _sim()
    sim.VIX_SIZING = True
    low = sim.vix_sizing({"regime": "LOW"})
    high = sim.vix_sizing({"regime": "EXTREME"})
    # low VIX → bigger size, nearer target; high VIX → smaller size, farther target
    assert low["qty_factor"] > high["qty_factor"]
    assert low["dist_mult"] < high["dist_mult"]
    assert sim._vix_regime({}) == "NORMAL"


def test_net_pnl_is_gross_minus_costs():
    sim = _sim()
    sim.INCLUDE_COSTS = True
    gross = sim.compute_pnl("BUY", 100.0, 103.0, 40)
    costs = sim.trade_costs(100.0, 103.0, 40, "BUY")
    net = round(gross - costs, 2)
    assert net < gross and costs > 0
