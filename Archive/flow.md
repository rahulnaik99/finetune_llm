# Architecture & Flow — llm_finetune

Mermaid diagrams of the trading-agent LLM fine-tuning pipeline. All top-down.

---

## 1. The flywheel loop (macro)

```mermaid
graph TD
    CFG[".env + config.py<br/>(all settings)"] --> FW["flywheel.py<br/>(orchestrator)"]
    FW -->|"1 simulate"| SIM["simulation.py<br/>TradingAgent"]
    SIM -->|"writes trades"| DB[("trading_agent.db<br/>trade_log")]
    FW -->|"2 build dataset"| DP["data_pipeline.py"]
    DB --> DP
    DP -->|"train/val/test JSONL"| DS[("data/rounds/round_NN")]
    FW -->|"3 fine-tune"| TR["train.py<br/>(MLX QLoRA)"]
    DS --> TR
    TR -->|"adapters + fused_model"| AD[("output/adapters/round_NN")]
    FW -->|"4 evaluate"| EV["evaluate.py<br/>evaluate_round()"]
    DS --> EV
    AD --> EV
    EV -->|"metrics"| LOG[("flywheel_log.json<br/>+ MLflow")]
    AD -.->|"next round decider<br/>(deploy loop)"| SIM
    FW -->|"repeat ROUNDS times"| FW
```

---

## 2. Inside one simulation episode (micro)

```mermaid
graph TD
    START["pick as_of date<br/>(no look-ahead)"] --> RAW["_get_raw_history()<br/>read CSV once, cache"]
    RAW --> DAILY["daily candles<br/>get_candles()"]
    RAW --> INTRA["15m + 1h candles<br/>get_intraday()"]

    DAILY --> TECH["compute_technicals()<br/>EMA50/100/200, VWAP,<br/>S/R zones, HH/HL/LH/LL"]
    DAILY --> LVL["_level_breaks()<br/>PDH/PDL, 2D, weekly"]
    INTRA --> ZN["compute_zones() + zones.py<br/>demand/supply 15m & 1h<br/>+ confluence"]
    VIXCSV["VIX CSV"] --> VIX["vix_context()<br/>level, pctile, regime"]
    XBRL["XBRL files"] --> FND["get_fundamentals()<br/>QoQ/YoY/TTM"]

    TECH --> PB["compact_prompt.py<br/>build_compact_user()"]
    LVL --> PB
    ZN --> PB
    VIX --> PB
    FND --> PB
    HIST["trade history<br/>(from DB)"] --> PB

    PB -->|"HEADLINE + REL +<br/>EMA/levels/zones/VIX/fund"| DEC["pre_trade_decision()"]
    DEC --> BK{{"LLM backend?"}}
    BK -->|"ollama"| OLL["Ollama teacher<br/>nemotron-bolt-e4b-16k"]
    BK -->|"adapter"| STU["fine-tuned student<br/>(MLX)"]
    BK -->|"fallback"| RULE["rule-based logic"]
    OLL --> JSON["decision JSON<br/>action/SL/TP/RR/<br/>reasoning/features_used"]
    STU --> JSON
    RULE --> JSON

    JSON --> FWD["get_forward_candles()<br/>+ check_sl_tp()"]
    FWD --> PNL["compute_pnl()<br/>exit reason, P&L"]
    PNL --> POST["post_trade_analysis()<br/>lesson_learned"]
    POST --> WRITE[("write row to<br/>trading_agent.db")]
    WRITE --> HIST
```

---

## 3. Prompt assembly (compact_prompt.py)

```mermaid
graph TD
    IN["tech + fund + history dicts"] --> H["HEADLINE:<br/>bias/trend/PDH/VIX/confluence"]
    IN --> R["REL: scale-free<br/>vsEMA200 %, ATR-mult to S/R"]
    IN --> P["Price block:<br/>C, 5d range, ATR, vol, body"]
    IN --> E["EMA stack + VWAP + Trend/Struct"]
    IN --> SR["S/R zones + swings"]
    IN --> L["LEVELS: PDH/PDL/2D/weekly states"]
    IN --> V["VIX: level/pctile/regime/dir"]
    IN --> Z["ZONES: 15m & 1h DZ/SZ + CONFLUENCE"]
    IN --> F["Fund: PAT/Rev/GNPA (YoY/QoQ)"]
    IN --> HI["HISTORY: last 5 trades + lessons"]
    H --> OUT["~188-token prompt"]
    R --> OUT
    P --> OUT
    E --> OUT
    SR --> OUT
    L --> OUT
    V --> OUT
    Z --> OUT
    F --> OUT
    HI --> OUT
```

---

## 4. Dataset build -> training -> evaluation

```mermaid
graph TD
    DB[("trading_agent.db")] --> LT["load_trades()"]
    LT --> BE["build_pre_trade_example()<br/>build_post_trade_example()<br/>(+ _meta: date, pnl)"]
    BE --> QF["quality_filter()<br/>min_confidence, min_rr,<br/>drop HOLD"]
    QF --> BD["balance_and_dedup()<br/>remove near-dupes,<br/>cap BUY/SELL 1.5x"]
    BD --> SP{{"SPLIT_MODE"}}
    SP -->|"time (walk-forward)"| TIME["sort by date<br/>train=early, test=latest"]
    SP -->|"random"| RND["shuffle"]
    TIME --> JL[("train/val/test.jsonl")]
    RND --> JL

    JL --> TRN["train.py<br/>hardware auto-detect"]
    TRN --> HW{{"device?"}}
    HW -->|"Apple Silicon"| MLX["MLX LoRA/QLoRA<br/>4-bit OptiQ"]
    HW -->|"NVIDIA >=16GB"| HF["HF LoRA"]
    HW -->|"NVIDIA small"| QL["QLoRA 4/8-bit"]
    MLX --> ADP[("adapters + fused_model")]
    HF --> ADP
    QL --> ADP

    JL --> EVAL["evaluate.py"]
    ADP --> EVAL
    EVAL --> MQ["model quality:<br/>JSON/schema/reasoning/<br/>features_used/action_match"]
    EVAL --> TM["trading metrics:<br/>win rate, expectancy,<br/>PF, Sharpe, maxDD"]
    MQ --> MF[("MLflow + flywheel_log")]
    TM --> MF
```

---

## 5. Teacher / student and the deploy loop

```mermaid
graph TD
    T["TEACHER<br/>nemotron-bolt-e4b-16k<br/>(Ollama, 16k ctx)"] -->|"generates decisions<br/>USE_OLLAMA=1"| DATA["training data"]
    DATA --> TRAIN["train.py (MLX QLoRA)"]
    BASE["STUDENT base<br/>Nemotron-3.5-Bolt-E4B-OptiQ-4bit"] --> TRAIN
    TRAIN --> FUSED["fused_model (MLX)"]
    FUSED -->|"deploy_to_ollama.py"| CONV["convert MLX -> GGUF"]
    CONV --> MODF["Modelfile<br/>num_ctx 16384"]
    MODF --> CREATE["ollama create<br/>bolt-e4b-16k-trading"]
    CREATE -.->|"becomes next decider"| T
    FUSED -.->|"or adapter backend"| STUD["student in-loop"]
    DATA -->|"outcomes"| DPO["dpo_sketch.py<br/>preference pairs"]
    DPO -->|"mlx_lm.dpo"| TRAIN
```
