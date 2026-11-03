"""
flywheel.py
───────────
Integrated data-generation + fine-tuning loop.

Each round:
  1. Run N simulation episodes  (uses current best model)
  2. Convert new data to JSONL  (data_pipeline.py logic)
  3. Fine-tune on accumulated data  (train_qlora.py logic)
  4. Swap the new adapter into the simulation
  5. Repeat

Usage:
    python flywheel.py \
        --symbol CANBK \
        --start_date 2024-01-10 \
        --rounds 5 \
        --episodes_per_round 50 \
        --adapter_dir output/adapters
"""

import argparse, json, os, shutil, sys
from datetime import datetime
from pathlib import Path

# ── make sure the simulation notebook code is importable ─────────────────────
# (assumes flywheel.py lives next to trading_agent_sim's helper functions,
#  or you copy the cell logic into separate .py modules)
# For a clean project layout, refactor each notebook cell into a module:
#   simulation/data.py         ← get_candles, get_fundamentals
#   simulation/technicals.py   ← compute_technicals
#   simulation/llm.py          ← pre_trade_decision, post_trade_analysis
#   simulation/agent.py        ← TradingAgent, check_sl_tp
# Then import them here.

ROUNDS_LOG = []   # tracks performance across rounds


# ════════════════════════════════════════════════════════════════════
#  STEP 1 — run simulation episodes with the CURRENT best model
# ════════════════════════════════════════════════════════════════════
def run_simulation_round(
    symbol: str,
    start_date: str,
    n_episodes: int,
    adapter_path: str | None,
    db_path: str,
    initial_cap: float = 10_000,
) -> dict:
    """
    Run N trading episodes.
    If adapter_path is set, use the fine-tuned model.
    Otherwise use FALLBACK_MODE (rule-based) for the very first round.
    """
    # Swap model source in the simulation
    if adapter_path and Path(adapter_path).exists():
        # Fine-tuned model available → use it
        from inference import get_trader
        trader = get_trader(adapter_path)

        # Monkey-patch the _ollama function used inside pre_trade_decision
        # NOTE: import simulation.llm as llm_module — convert notebook to module first
        original_ollama = llm_module._ollama

        def patched_ollama(system, user, temp=0.2):
            return trader.generate(system, user)

        llm_module._ollama = patched_ollama
        print(f"  Using fine-tuned adapter: {adapter_path}")
    else:
        print("  No adapter yet — using rule-based fallback")

    # Import TradingAgent (adjust import path to your layout)
    # NOTE: from simulation.agent import TradingAgent — convert notebook to module first

    agent = TradingAgent(symbol=symbol, start_date=start_date, capital=initial_cap)
    for ep in range(1, n_episodes + 1):
        if not agent.alive():
            print(f"  Agent died at episode {ep}")
            break
        agent.run_episode()

    # Restore original ollama function
    if adapter_path and Path(adapter_path).exists():
        llm_module._ollama = original_ollama

    wins   = sum(1 for t in agent.history if t["pnl"] > 0)
    losses = sum(1 for t in agent.history if t["pnl"] < 0)
    total_pnl = sum(t["pnl"] for t in agent.history)
    return {
        "episodes": agent.episode,
        "wins": wins, "losses": losses,
        "total_pnl": total_pnl,
        "final_capital": agent.capital,
        "win_rate": round(wins / max(agent.episode, 1), 2),
    }


# ════════════════════════════════════════════════════════════════════
#  STEP 2 — convert new SQLite data → JSONL
# ════════════════════════════════════════════════════════════════════
def build_dataset_for_round(db_path: str, data_dir: str, round_num: int) -> int:
    """Returns number of training examples generated."""
    from data_pipeline import build_dataset

    out_path = f"{data_dir}/round_{round_num:02d}/trading.jsonl"
    counts = build_dataset(
        db_path=db_path,
        out_path=out_path,
        min_confidence=0.5,
        min_rr=1.3,   # slightly relaxed per round so early bad data is still used
    )
    return counts.get("train", 0)


# ════════════════════════════════════════════════════════════════════
#  STEP 3 — fine-tune on ALL accumulated data so far
#           (not just this round — use the whole growing buffer)
# ════════════════════════════════════════════════════════════════════
def finetune_round(
    data_dir: str,
    round_num: int,
    adapter_dir: str,
    base_model: str = "nvidia/Nemotron-Nano-4B-Instruct",
    prev_adapter: str | None = None,
) -> str:
    """
    Merge all JSONL files from rounds 1..N into one train set.
    Fine-tune and save adapter to adapter_dir/round_N/.
    Returns path to new adapter.
    """
    import subprocess

    # Merge all training data so far
    all_train = []
    for r in range(1, round_num + 1):
        f = Path(data_dir) / f"round_{r:02d}" / "trading_train.jsonl"
        if f.exists():
            with open(f) as fh:
                all_train.extend(fh.readlines())

    merged_path = Path(data_dir) / "merged_train.jsonl"
    with open(merged_path, "w") as fh:
        fh.writelines(all_train)

    val_path = Path(data_dir) / f"round_{round_num:02d}" / "trading_val.jsonl"

    new_adapter = Path(adapter_dir) / f"round_{round_num:02d}"
    new_adapter.mkdir(parents=True, exist_ok=True)

    # Start from previous adapter if available (continual learning)
    start_from = str(prev_adapter) if prev_adapter and Path(prev_adapter).exists() else base_model

    cmd = [
        sys.executable, "train_qlora.py",
        "--train",      str(merged_path),
        "--val",        str(val_path) if val_path.exists() else str(merged_path),
        "--model",      start_from,
        "--output_dir", str(new_adapter),
        "--epochs",     "2",          # fewer epochs per round (data grows each round)
        "--lora_r",     "16",
    ]

    print(f"  Training on {len(all_train)} total examples...")
    print(f"  Starting from: {start_from}")
    result = subprocess.run(cmd, capture_output=False)

    final = new_adapter / "final_adapter"
    if not final.exists():
        raise RuntimeError(f"Training failed — adapter not found at {final}")

    print(f"  New adapter: {final}")
    return str(final)


# ════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════
def flywheel(
    symbol: str,
    start_date: str,
    n_rounds: int,
    episodes_per_round: int,
    adapter_dir: str,
    data_dir: str,
    db_path: str,
    base_model: str,
):
    current_adapter = None   # None = use fallback for round 1

    print(f"\n{'='*70}")
    print(f"  FLYWHEEL: {symbol}  start={start_date}")
    print(f"  {n_rounds} rounds × {episodes_per_round} episodes = {n_rounds*episodes_per_round} total")
    print(f"{'='*70}\n")

    for rnd in range(1, n_rounds + 1):
        print(f"\n{'─'*70}")
        print(f"  ROUND {rnd}/{n_rounds}")
        print(f"{'─'*70}")

        # ── 1. Run simulation ────────────────────────────────────────────────
        print(f"\n[1/3] Simulation — {episodes_per_round} episodes with "
              f"{'adapter v'+str(rnd-1) if current_adapter else 'rule-based fallback'}")
        sim_stats = run_simulation_round(
            symbol=symbol,
            start_date=start_date,
            n_episodes=episodes_per_round,
            adapter_path=current_adapter,
            db_path=db_path,
        )
        print(f"  Win rate: {sim_stats['win_rate']:.0%}  "
              f"P&L: {sim_stats['total_pnl']:+.0f}  "
              f"Final cap: {sim_stats['final_capital']:,.0f}")

        # ── 2. Build dataset ─────────────────────────────────────────────────
        print(f"\n[2/3] Building dataset...")
        n_examples = build_dataset_for_round(db_path, data_dir, rnd)
        print(f"  {n_examples} training examples this round")

        if n_examples < 10:
            print("  Too few examples — skipping training this round")
            ROUNDS_LOG.append({"round": rnd, "sim": sim_stats, "trained": False})
            continue

        # ── 3. Fine-tune ─────────────────────────────────────────────────────
        print(f"\n[3/3] Fine-tuning...")
        try:
            new_adapter = finetune_round(
                data_dir=data_dir,
                round_num=rnd,
                adapter_dir=adapter_dir,
                base_model=base_model,
                prev_adapter=current_adapter,
            )
            current_adapter = new_adapter
            print(f"  Adapter updated → {new_adapter}")
        except Exception as e:
            print(f"  Training failed: {e}. Keeping previous adapter.")

        ROUNDS_LOG.append({
            "round": rnd,
            "sim": sim_stats,
            "n_examples": n_examples,
            "adapter": current_adapter,
            "trained": True,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  FLYWHEEL COMPLETE — {n_rounds} rounds")
    print(f"  {'Round':<8} {'Win%':<8} {'P&L':<10} {'Examples':<10} {'Trained'}")
    print(f"  {'─'*50}")
    for log in ROUNDS_LOG:
        s = log["sim"]
        print(f"  Round {log['round']:<3} "
              f"{s['win_rate']:.0%}     "
              f"{s['total_pnl']:+8.0f}   "
              f"{log.get('n_examples',0):<10} "
              f"{'✓' if log.get('trained') else '✗'}")
    print(f"{'='*70}")
    print(f"  Final adapter: {current_adapter}")

    # Save log
    log_path = Path(adapter_dir) / "flywheel_log.json"
    with open(log_path, "w") as f:
        json.dump(ROUNDS_LOG, f, indent=2, default=str)
    print(f"  Log saved: {log_path}")
    return current_adapter


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",             default="CANBK")
    p.add_argument("--start_date",         default="2024-01-10")
    p.add_argument("--rounds",       type=int, default=5)
    p.add_argument("--episodes",     type=int, default=50)
    p.add_argument("--adapter_dir",        default="output/adapters")
    p.add_argument("--data_dir",           default="data/rounds")
    p.add_argument("--db",                 default="trading_agent.db")
    p.add_argument("--base_model",         default="nvidia/Nemotron-Nano-4B-Instruct")
    args = p.parse_args()

    flywheel(
        symbol=args.symbol,
        start_date=args.start_date,
        n_rounds=args.rounds,
        episodes_per_round=args.episodes,
        adapter_dir=args.adapter_dir,
        data_dir=args.data_dir,
        db_path=args.db,
        base_model=args.base_model,
    )
