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

# ── The simulation now lives in simulation.py (importable module) ────────────
# All simulation logic — data fetching, technicals, LLM decisions, TradingAgent —
# is in simulation.py. Both this file and trading_agent_sim.ipynb import from it.
import simulation
from simulation import TradingAgent, init_db, set_llm_backend

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
    If adapter_path is set, use the fine-tuned model (student).
    Otherwise use the teacher model (Ollama) if set, else rule-based fallback.
    """
    simulation.DB_PATH = db_path
    init_db()

    if adapter_path and Path(adapter_path).exists():
        set_llm_backend("adapter", adapter_path)
        print(f"  Using fine-tuned adapter: {adapter_path}")
    elif os.environ.get("USE_OLLAMA", "0") == "1":
        set_llm_backend("ollama")
        print(f"  Using Ollama teacher model: {simulation.OLLAMA_MODEL}")
    else:
        set_llm_backend("fallback")
        print("  Using rule-based fallback (set USE_OLLAMA=1 for teacher model)")

    agent = TradingAgent(symbol=symbol, start_date=start_date, capital=initial_cap)
    for ep in range(1, n_episodes + 1):
        if not agent.alive():
            print(f"  Agent died at episode {ep} (capital {agent.capital:,.0f})")
            break
        try:
            agent.run_episode()
        except Exception as e:
            print(f"  Episode {ep} error: {e} -- skipping day")
            agent.date = simulation.next_trading_day(agent.date, 1)

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

    # Use train.py — it auto-detects hardware (MLX on Apple Silicon, QLoRA on NVIDIA).
    # This avoids needing torch/bitsandbytes on a Mac.
    # Iterations, rank and num-layers come from .env so you never edit code.
    from config import env_int
    cmd = [
        sys.executable, "train.py",
        "--train",      str(merged_path),
        "--val",        str(val_path) if val_path.exists() else str(merged_path),
        "--model",      start_from,
        "--output_dir", str(new_adapter),
        "--epochs",     str(env_int("EPOCHS", 2)),
        "--lora_r",     str(env_int("LORA_R", 16)),
        "--iters",      str(env_int("MLX_ITERS", 600)),
        "--save_every", str(env_int("SAVE_EVERY", 100)),
    ]

    print(f"  Training on {len(all_train)} total examples...")
    print(f"  Starting from: {start_from}")
    print(f"  (train.py auto-detects hardware: MLX on Apple Silicon, QLoRA on NVIDIA)")
    result = subprocess.run(cmd, capture_output=False)

    # train.py (MLX path) saves to <output_dir>/adapters or <output_dir>/final_adapter
    candidates = [
        new_adapter / "final_adapter",
        new_adapter / "adapters",
        new_adapter / "fused_model",
    ]
    final = next((c for c in candidates if c.exists()), None)
    if final is None:
        raise RuntimeError(
            f"Training failed — no adapter found in {new_adapter}. "
            f"On Mac: pip install mlx-lm. On NVIDIA: pip install torch peft trl bitsandbytes."
        )

    print(f"  New adapter: {final}")
    return str(final)


# ════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════
def evaluate_round(data_dir: str, round_num: int, adapter_path: str | None) -> dict:
    """
    Evaluate the round on its walk-forward TEST split:
      - trading metrics from recorded P&L (pure math, always available)
      - model-quality metrics IF a trained adapter + MLX are available (optional)
    Prints a summary, logs to MLflow when configured, and never raises.
    """
    out = {}
    try:
        from evaluate import trading_metrics, pnls_from_test_meta, evaluate_model_quality
        test_path = Path(data_dir) / f"round_{round_num:02d}" / "trading_test.jsonl"
        if not test_path.exists():
            return out

        tm = trading_metrics(pnls_from_test_meta(str(test_path)))
        out["trading"] = tm
        print(f"\n[4/4] Eval (walk-forward test): win {tm.get('win_rate','?')}% | "
              f"expectancy {tm.get('expectancy','?')} | PF {tm.get('profit_factor','?')} | "
              f"Sharpe {tm.get('sharpe_per_trade','?')} | maxDD {tm.get('max_drawdown','?')}")

        # Optional model-quality (needs the fine-tuned model + mlx_lm)
        fused = None
        if adapter_path:
            cand = [Path(adapter_path), Path(adapter_path).parent / "fused_model"]
            fused = next((str(c) for c in cand if c.exists()), None)
        if fused:
            try:
                from evaluate import mlx_generate_factory
                gen = mlx_generate_factory(fused)
                q = evaluate_model_quality(str(test_path), gen)
                out["quality"] = q
                print(f"       quality: JSON {q['json_validity']}% | schema {q['schema_ok']}% | "
                      f"reasoning {q['reasoning_ok']}% | features_used {q['features_used_ok']}% | "
                      f"action_match {q['action_match']}%")
            except Exception as e:
                print(f"       (model-quality eval skipped: {e})")

        # Log to MLflow if available
        try:
            import mlflow
            with mlflow.start_run(run_name=f"eval-round-{round_num}", nested=True):
                for k, v in (out.get("trading") or {}).items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"test_{k}", v)
                for k, v in (out.get("quality") or {}).items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"quality_{k}", v)
        except Exception:
            pass
    except Exception as e:
        print(f"  (eval skipped: {e})")
    return out


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

        # ── 4. Evaluate on the walk-forward TEST set (always-on, never fatal) ──
        eval_metrics = evaluate_round(data_dir, rnd, current_adapter)

        ROUNDS_LOG.append({
            "round": rnd,
            "sim": sim_stats,
            "n_examples": n_examples,
            "adapter": current_adapter,
            "trained": True,
            "eval": eval_metrics,
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
    log_path.parent.mkdir(parents=True, exist_ok=True)   # ensure dir exists even if no training ran
    with open(log_path, "w") as f:
        json.dump(ROUNDS_LOG, f, indent=2, default=str)
    print(f"  Log saved: {log_path}")
    return current_adapter


if __name__ == "__main__":
    # Load .env so every knob (symbol, rounds, episodes, model, backend...) is
    # configurable WITHOUT editing code. CLI flags still override .env.
    from config import load_env, env_str, env_int, env_bool
    load_env()

    # USE_OLLAMA / OLLAMA_MODEL are read by simulation via the environment;
    # make sure .env values are visible to run_simulation_round's checks.
    p = argparse.ArgumentParser()
    p.add_argument("--symbol",       default=env_str("SYMBOL", "CANBK"))
    p.add_argument("--start_date",   default=env_str("START_DATE", "2024-01-10"))
    p.add_argument("--rounds",       type=int, default=env_int("ROUNDS", 5))
    p.add_argument("--episodes",     type=int, default=env_int("EPISODES", 50))
    p.add_argument("--adapter_dir",  default=env_str("ADAPTER_DIR", "output/adapters"))
    p.add_argument("--data_dir",     default=env_str("DATA_DIR", "data/rounds"))
    p.add_argument("--db",           default=env_str("DB_PATH", "trading_agent.db"))
    p.add_argument("--base_model",   default=env_str("BASE_MODEL",
                                                     "mlx-community/NVIDIA-Nemotron-3.5-Bolt-E4B-OptiQ-4bit"))
    args = p.parse_args()

    print("="*70)
    print("  CONFIG (from .env / CLI)")
    print(f"    symbol={args.symbol}  start={args.start_date}  "
          f"rounds={args.rounds}  episodes={args.episodes}")
    print(f"    base_model={args.base_model}")
    print(f"    USE_OLLAMA={env_bool('USE_OLLAMA', False)}  "
          f"OLLAMA_MODEL={env_str('OLLAMA_MODEL', 'nemotron-bolt-e4b-16k')}")
    print(f"    mlx_iters={env_int('MLX_ITERS', 600)}  lora_r={env_int('LORA_R', 16)}")
    print("="*70)

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
