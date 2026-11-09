"""
simulation.py
─────────────
The trading-agent simulation as an importable Python module.

Previously this logic lived only inside trading_agent_sim.ipynb (notebook cells),
so flywheel.py could not import TradingAgent. This module is the single source of
truth — both the notebook and flywheel.py import from here.

Public API:
    init_db()
    get_candles(symbol, as_of_date, lookback_days)
    get_fundamentals(symbol, as_of_date)
    compute_technicals(df)
    TradingAgent(symbol, start_date, capital)
    set_llm_backend("fallback" | "ollama" | "adapter", adapter_path=None)

Usage in flywheel.py:
    from simulation import TradingAgent, init_db, set_llm_backend
    init_db()
    set_llm_backend("ollama")                    # or "fallback" or "adapter"
    agent = TradingAgent("CANBK", "2024-01-10")
    while agent.alive() and agent.episode < 50:
        agent.run_episode()
"""
from __future__ import annotations

import json, sqlite3, re, textwrap, os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import warnings
# Silence the flood of pandas/yfinance deprecation warnings (inplace, PDEP-8, etc.)
# These come from yfinance internals, not our code — safe to suppress.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings("ignore", category=Pandas4Warning)
except Exception:
    pass

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ── Load .env BEFORE reading any config below ───────────────────────────────
# flywheel.py imports this module before it calls load_env(), so we must apply
# .env here or every setting below would silently fall back to its default
# (DATA_SOURCE=yfinance, FUNDAMENTALS_SOURCE=yfinance, VIX off, ...).
try:
    from config import load_env
    load_env()
except Exception:
    pass


# ════════════════════════════════════════════════════════════════════
#  CONFIG (overridable via environment variables)
# ════════════════════════════════════════════════════════════════════
INITIAL_CAP     = float(os.environ.get("INITIAL_CAP", 10_000))
LOOKBACK_DAYS   = int(os.environ.get("LOOKBACK_DAYS", 30))
MAX_HOLD_DAYS   = int(os.environ.get("MAX_HOLD_DAYS", 5))
DEATH_THRESHOLD = float(os.environ.get("DEATH_THRESHOLD", 0.20))
RISK_PER_TRADE  = float(os.environ.get("RISK_PER_TRADE", 0.02))
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "nemotron-bolt-e4b-16k")
DB_PATH         = os.environ.get("DB_PATH", "trading_agent.db")

# ── Price data source: "csv" (local file, offline) or "yfinance" (online) ──
DATA_SOURCE = os.environ.get("DATA_SOURCE", "yfinance").strip().lower()
CSV_DIR     = os.environ.get("CSV_DIR", "data/prices")   # looks for {CSV_DIR}/{SYMBOL}.csv
CSV_PATH    = os.environ.get("CSV_PATH", "")             # explicit single file (overrides CSV_DIR)

# ── Fundamentals source: "xbrl" (official SEBI filings) or "yfinance" ───────
FUNDAMENTALS_SOURCE = os.environ.get("FUNDAMENTALS_SOURCE", "yfinance").strip().lower()
XBRL_DIR            = os.environ.get("XBRL_DIR", "data/fundamentals")  # {XBRL_DIR}/{SYMBOL}/*.xml

# ── India VIX (volatility regime context) — optional local CSV ──────────────
VIX_CSV      = os.environ.get("VIX_CSV", "")          # e.g. data/prices/INDIAVIX.csv (no volume needed)
VIX_LOW      = float(os.environ.get("VIX_LOW", 13))   # regime thresholds (India VIX)
VIX_NORMAL   = float(os.environ.get("VIX_NORMAL", 18))
VIX_ELEVATED = float(os.environ.get("VIX_ELEVATED", 25))


# ════════════════════════════════════════════════════════════════════
#  LLM BACKEND SWITCH
# ════════════════════════════════════════════════════════════════════
_BACKEND = {"mode": "fallback", "adapter_path": None, "trader": None}


def set_llm_backend(mode: str, adapter_path: str | None = None) -> None:
    """
    Choose how trade decisions are generated:
      "fallback" — rule-based (no LLM, deterministic, for Round 1)
      "ollama"   — Ollama server (teacher model)
      "adapter"  — fine-tuned MLX/HF adapter (student model, loaded lazily)
    """
    _BACKEND["mode"] = mode
    _BACKEND["adapter_path"] = adapter_path
    _BACKEND["trader"] = None   # reset lazy-loaded model
    print(f"  LLM backend set to: {mode}" + (f" ({adapter_path})" if adapter_path else ""))


def _llm_generate(system: str, user: str, temp: float = 0.2) -> str | None:
    """Route to the active backend. Returns raw text or None (→ caller uses fallback)."""
    mode = _BACKEND["mode"]

    if mode == "fallback":
        return None   # signals caller to use rule-based

    if mode == "ollama":
        try:
            r = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system},
                             {"role": "user",   "content": user}],
                "stream": False,
                "options": {"temperature": temp, "num_ctx": 8192},
            }, timeout=180)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            print(f"    [ollama error] {e} → fallback")
            return None

    if mode == "adapter":
        # Lazy-load the fine-tuned model once
        if _BACKEND["trader"] is None:
            try:
                from inference_mlx import generate_mlx
                _BACKEND["trader"] = ("mlx", _BACKEND["adapter_path"])
            except ImportError:
                try:
                    from inference import get_trader
                    _BACKEND["trader"] = ("hf", get_trader(_BACKEND["adapter_path"]))
                except Exception as e:
                    print(f"    [adapter load error] {e} → fallback")
                    return None
        kind, obj = _BACKEND["trader"]
        try:
            if kind == "mlx":
                from inference_mlx import generate_mlx
                return generate_mlx(obj, system, user)
            else:
                return obj.generate(system, user)
        except Exception as e:
            print(f"    [adapter error] {e} → fallback")
            return None

    return None


def _parse_json(text: str | None) -> dict:
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        return json.loads(text.strip())
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ════════════════════════════════════════════════════════════════════
#  DATABASE
# ════════════════════════════════════════════════════════════════════
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS fundamental_cache (
                symbol TEXT, as_of_quarter TEXT, data_json TEXT, fetched_at TEXT,
                PRIMARY KEY (symbol, as_of_quarter));
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode INTEGER, symbol TEXT, decision_date TEXT, action TEXT,
                entry_price REAL, quantity INTEGER, stop_loss REAL, take_profit REAL,
                exit_price REAL, exit_date TEXT, exit_reason TEXT, pnl REAL,
                capital_before REAL, capital_after REAL,
                pre_trade_reasoning TEXT, post_trade_analysis TEXT,
                lesson_learned TEXT, what_went_wrong TEXT,
                technicals_json TEXT, created_at TEXT);
        """)
    print(f"  DB ready: {DB_PATH}")


# ════════════════════════════════════════════════════════════════════
#  DATA FETCHER — strict as_of cutoff (MultiIndex-safe)
#  Downloads each symbol's FULL history ONCE and caches it, then slices
#  per episode. This avoids yfinance rate-limiting ("possibly delisted"
#  empty responses) that happens when you call download() 100s of times.
# ════════════════════════════════════════════════════════════════════
import time as _time

_PRICE_CACHE: dict[str, pd.DataFrame] = {}


def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def _download_with_retry(ticker: str, start: str, end: str, retries: int = 4) -> pd.DataFrame:
    """yfinance download with exponential backoff — handles transient rate limits."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d",
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                return df
            last_err = "empty response"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            wait = 2 ** attempt  # 2s, 4s, 8s
            print(f"    [yfinance retry {attempt}/{retries} for {ticker}: {last_err} - waiting {wait}s]")
            _time.sleep(wait)
    raise ValueError(f"No candle data for {ticker} after {retries} tries ({last_err})")


def _load_csv_raw(path: Path) -> pd.DataFrame:
    """
    Load a CSV into a timestamped OHLCV DataFrame (NOT resampled). Works for
    1-minute or daily files. Volume optional (indices like VIX have none).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}. Set CSV_PATH/CSV_DIR or DATA_SOURCE=yfinance."
        )
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    dt_col = next((c for c in df.columns if c in ("datetime", "date", "time", "timestamp")), None)
    if dt_col is None:
        dt_col = next((c for c in df.columns if ("date" in c or "time" in c)), df.columns[0])
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce", dayfirst=True, format="mixed")
    df = df.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
    rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
              "vol": "volume", "adj close": "close", "adj_close": "close"}
    df = df.rename(columns=rename)
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {path} missing columns {missing}. Found: {list(df.columns)}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[need + ["volume"]].apply(pd.to_numeric, errors="coerce").dropna(subset=need)
    df["volume"] = df["volume"].fillna(0.0)
    return df


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = df.resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    return out


_RAW_CACHE: dict[str, pd.DataFrame] = {}


def _get_raw_history(symbol: str) -> pd.DataFrame:
    """Read the symbol's CSV ONCE (raw timestamps, minute or daily), cache it."""
    if symbol in _RAW_CACHE:
        return _RAW_CACHE[symbol]
    path = Path(CSV_PATH) if CSV_PATH else Path(CSV_DIR) / f"{symbol}.csv"
    raw = _load_csv_raw(path)
    _RAW_CACHE[symbol] = raw
    print(f"  Loaded {len(raw)} raw bars for {symbol} from {path}")
    return raw


def _load_csv_history(symbol: str) -> pd.DataFrame:
    """Daily OHLCV for a symbol, derived from the raw cache (minute or daily)."""
    daily = _resample_ohlcv(_get_raw_history(symbol), "1D")
    daily.index = pd.to_datetime(daily.index.date)   # date-at-midnight for exact as_of slicing
    return daily


def get_intraday(symbol: str, as_of_date: str, rule: str, n_bars: int = 150) -> pd.DataFrame:
    """
    Resample the raw minute data to `rule` (e.g. '15min', '1h'), keep only bars
    on/before as_of_date, return the last n_bars. Empty if the source is daily.
    """
    raw = _get_raw_history(symbol)
    cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(days=1)   # include the whole as_of day
    sliced = raw[raw.index < cutoff]
    if sliced.empty:
        return pd.DataFrame()
    tf = _resample_ohlcv(sliced, rule)
    return tf.tail(n_bars)


def _get_full_history(symbol: str) -> pd.DataFrame:
    """Load the symbol's entire DAILY history ONCE (CSV or yfinance), cache, reuse."""
    if symbol in _PRICE_CACHE:
        return _PRICE_CACHE[symbol]

    if DATA_SOURCE == "csv":
        df = _load_csv_history(symbol)
        src = f"CSV ({CSV_PATH or Path(CSV_DIR)/ (symbol+'.csv')})"
    else:
        df = _download_with_retry(f"{symbol}.NS", "2015-01-01",
                                  (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        df = _flatten_cols(df)
        src = "yfinance (single download)"

    _PRICE_CACHE[symbol] = df
    print(f"  Cached {len(df)} daily bars for {symbol} from {src}")
    return df


def get_candles(symbol: str, as_of_date: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    hist = _get_full_history(symbol)
    df = hist[hist.index <= pd.Timestamp(as_of_date)].tail(lookback_days).copy()
    if df.empty:
        raise ValueError(f"No candle data for {symbol} on/before {as_of_date}")
    return df


def get_forward_candles(symbol: str, from_date: str, n: int = MAX_HOLD_DAYS) -> pd.DataFrame:
    hist = _get_full_history(symbol)
    df = hist[hist.index > pd.Timestamp(from_date)].head(n).copy()
    return df


def compute_zones(symbol: str, as_of_date: str) -> dict:
    """
    Detect demand/supply zones on 15-min (short) and 1-hour (big) timeframes
    from the raw minute data, no look-ahead. Returns {} if the source is daily
    (no intraday data). Confluence = a 15m zone sitting inside a 1h zone.
    """
    try:
        from zones import detect_zones, nearest_zones, confluence
    except Exception:
        return {}
    tf15 = get_intraday(symbol, as_of_date, "15min", 160)
    tf1h = get_intraday(symbol, as_of_date, "1h", 160)
    if (tf15 is None or tf15.empty) and (tf1h is None or tf1h.empty):
        return {}   # daily-only source — no intraday zones
    price = float((tf15 if not tf15.empty else tf1h)["close"].iloc[-1])
    near15 = nearest_zones(detect_zones(tf15, impulse_mult=3.0), price) if not tf15.empty else {}
    near1h = nearest_zones(detect_zones(tf1h, impulse_mult=3.0), price) if not tf1h.empty else {}
    return {"tf_15m": near15, "tf_1h": near1h, "confluence": confluence(near15, near1h)}


def _quarter_key(date_str: str) -> str:
    t = pd.Timestamp(date_str)
    return f"{t.year}-Q{(t.month - 1)//3 + 1}"


def _fund_completeness(fund: dict, as_of: str) -> str:
    """
    Context-completeness tag for a trade's fundamentals:
      none  = no fundamentals available at all
      fresh = most-recent filing reported within ~140 days (normal quarterly cadence)
      stale = carried forward > ~140 days (a quarter is missing from our data)
    """
    if not fund:
        return "none"
    ref = fund.get("result_date") or fund.get("quarter_end")
    if not ref:
        return "stale"
    try:
        days = (pd.Timestamp(as_of) - pd.Timestamp(ref)).days
    except Exception:
        return "stale"
    return "fresh" if 0 <= days <= 140 else "stale"


def get_fundamentals(symbol: str, as_of_date: str) -> dict:
    # ── XBRL source (official SEBI filings) — preferred when configured ──────
    if FUNDAMENTALS_SOURCE == "xbrl":
        try:
            from fundamentals_xbrl import get_fundamentals_xbrl
            f = get_fundamentals_xbrl(symbol, as_of_date, XBRL_DIR)
            if f:
                # map to the keys the prompt builder expects (+ keep the rich ones)
                f.setdefault("net_income", f.get("pat"))
                f.setdefault("quarter_key", _quarter_key(f.get("quarter_end", as_of_date)))
                return f
            # No XBRL filing on/before this date. Fundamentals are OPTIONAL
            # (15-year training), so return empty → tagged 'none'. Do NOT fall
            # back to yfinance here (it crashes on 'N/A' for old dates and would
            # mix sources). Set FUNDAMENTALS_SOURCE=yfinance explicitly to use it.
            return {}
        except Exception as e:
            print(f"  [xbrl error: {e}] — continuing without fundamentals")
            return {}

    qkey = _quarter_key(as_of_date)
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT data_json FROM fundamental_cache WHERE symbol=? AND as_of_quarter=?",
                          (symbol, qkey)).fetchone()
    if row:
        return json.loads(row[0])

    tk = yf.Ticker(f"{symbol}.NS")
    cutoff = pd.Timestamp(as_of_date)
    data = {"symbol": symbol, "as_of": as_of_date, "quarter_key": qkey}

    def safe(df, name, col):
        try:
            v = df.loc[name, col]
            return float(v) if pd.notna(v) else None
        except Exception:
            return None

    try:
        qf = tk.quarterly_financials
        if qf is not None and not qf.empty:
            past = [c for c in qf.columns if c <= cutoff]
            if past:
                col = past[0]
                data["quarter_end"] = str(col.date())
                for k in ["Total Revenue", "Net Income", "Operating Income", "Basic EPS"]:
                    data[k.lower().replace(" ", "_")] = safe(qf, k, col)
    except Exception as e:
        data["err_income"] = str(e)[:100]

    try:
        info = tk.info or {}
        for k in ["trailingPE", "priceToBook", "returnOnEquity"]:
            data[k] = info.get(k)
    except Exception:
        pass

    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT OR REPLACE INTO fundamental_cache VALUES (?,?,?,?)",
                    (symbol, qkey, json.dumps(data), datetime.now().isoformat()))
    return data


# ════════════════════════════════════════════════════════════════════
#  TECHNICAL ENGINE
# ════════════════════════════════════════════════════════════════════
def _detect_pivots(series: pd.Series, order: int = 2):
    pivots = []
    for i in range(order, len(series) - order):
        win = series.iloc[i - order:i + order + 1]
        if series.iloc[i] == win.max():
            pivots.append(("high", series.index[i], float(series.iloc[i])))
        elif series.iloc[i] == win.min():
            pivots.append(("low", series.index[i], float(series.iloc[i])))
    return pivots


def _classify_structure(pivots):
    sh = sorted([(d, p) for t, d, p in pivots if t == "high"])
    sl = sorted([(d, p) for t, d, p in pivots if t == "low"])
    if len(sh) < 2 or len(sl) < 2:
        return "ranging", []
    hp = [p for _, p in sh[-3:]]; lp = [p for _, p in sl[-3:]]
    hh = all(a < b for a, b in zip(hp, hp[1:])); hl = all(a < b for a, b in zip(lp, lp[1:]))
    lh = all(a > b for a, b in zip(hp, hp[1:])); ll = all(a > b for a, b in zip(lp, lp[1:]))
    if hh and hl: return "uptrend", ["HH", "HL"]
    if lh and ll: return "downtrend", ["LH", "LL"]
    if hh and ll: return "expanding", ["HH", "LL"]
    if lh and hl: return "contracting", ["LH", "HL"]
    return "ranging", []


def _cluster_zones(pivots, kind, bw=0.008):
    prices = sorted([p for t, _, p in pivots if t == kind])
    if not prices:
        return []
    zones, c = [], [prices[0]]
    for p in prices[1:]:
        if (p - c[-1]) / c[-1] < bw:
            c.append(p)
        else:
            zones.append((round(min(c), 2), round(max(c), 2), round(float(np.mean(c)), 2)))
            c = [p]
    zones.append((round(min(c), 2), round(max(c), 2), round(float(np.mean(c)), 2)))
    return zones


_VIX_CACHE = {}


def _get_vix_history():
    """Load the India VIX daily series from VIX_CSV once, cached. None if unset/missing."""
    if not VIX_CSV:
        return None
    if "vix" in _VIX_CACHE:
        return _VIX_CACHE["vix"]
    try:
        df = _load_vix_csv(Path(VIX_CSV))
        _VIX_CACHE["vix"] = df
        print(f"  Cached {len(df)} daily VIX bars from {VIX_CSV}")
    except Exception as e:
        print(f"  [VIX load failed: {e}] — continuing without VIX")
        df = None
        _VIX_CACHE["vix"] = None
    return df


def _load_vix_csv(path: Path) -> pd.DataFrame:
    """Load a VIX CSV (no volume needed) → daily close series. Reuses the OHLCV loader rules."""
    global CSV_PATH
    _saved = CSV_PATH
    CSV_PATH = str(path)
    try:
        return _load_csv_history("__VIX__")   # CSV_PATH override makes symbol irrelevant
    finally:
        CSV_PATH = _saved


def vix_context(as_of: str) -> dict:
    """
    VIX regime context as of a date (no look-ahead). Returns {} if no VIX data.
    Fields: level, 1y percentile, regime bucket, 5-day direction, 20-day range position.
    """
    hist = _get_vix_history()
    if hist is None or hist.empty:
        return {}
    s = hist["close"][hist.index <= pd.Timestamp(as_of)]
    if len(s) < 2:
        return {}
    level = float(s.iloc[-1])
    yr = s.tail(252)
    pctile = round(float((yr < level).mean()) * 100)
    win20 = s.tail(20)
    lo, hi = float(win20.min()), float(win20.max())
    pos20 = ("top" if hi > lo and (level - lo) / (hi - lo) > 0.66 else
             "bottom" if hi > lo and (level - lo) / (hi - lo) < 0.33 else "mid")
    chg5 = level - float(s.iloc[-6]) if len(s) >= 6 else 0.0
    direction = "rising" if chg5 > 0.3 else "falling" if chg5 < -0.3 else "flat"
    regime = ("LOW" if level < VIX_LOW else
              "NORMAL" if level < VIX_NORMAL else
              "ELEVATED" if level < VIX_ELEVATED else "EXTREME")
    return {"vix": round(level, 2), "pctile_1y": pctile, "regime": regime,
            "dir_5d": direction, "pos_20d": pos20}


def compute_technicals(df: pd.DataFrame) -> dict:
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    cur = float(close.iloc[-1])
    e50  = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    e100 = float(close.ewm(span=100, adjust=False).mean().iloc[-1])
    e200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
    tp   = (high + low + close) / 3
    vwap = float((tp * vol).cumsum().iloc[-1] / vol.cumsum().iloc[-1])

    pivots = _detect_pivots(high) + _detect_pivots(low)
    trend, structure = _classify_structure(pivots)
    sup = [(l, h, m) for l, h, m in _cluster_zones(pivots, "low") if m < cur]
    res = [(l, h, m) for l, h, m in _cluster_zones(pivots, "high") if m > cur]
    ns = max(sup, key=lambda z: z[2]) if sup else None
    nr = min(res, key=lambda z: z[2]) if res else None

    def zf(z): return {"low": z[0], "high": z[1], "mid": z[2]} if z else None

    stack = ("bullish (50>100>200)" if e50 > e100 > e200 else
             "bearish (50<100<200)" if e50 < e100 < e200 else "mixed")
    recent = (df.tail(5).reset_index()
              .rename(columns={"Date": "date", "index": "date"})
              [["date", "open", "high", "low", "close", "volume"]]
              .assign(date=lambda d: d["date"].astype(str)).round(2)
              .to_dict(orient="records"))

    # ── Level-break features (PDH/PDL, 2-day, weekly) + broke/held flags ────
    levels = _level_breaks(df, cur)

    return {
        "close": cur, "ema_50": round(e50, 2), "ema_100": round(e100, 2),
        "ema_200": round(e200, 2), "vwap": round(vwap, 2), "ema_stack": stack,
        "above_vwap": cur > vwap, "above_ema200": cur > e200,
        "trend": trend, "structure": structure,
        "nearest_support": zf(ns), "nearest_resistance": zf(nr),
        "support_zones": [zf(z) for z in sup[-3:]],
        "resistance_zones": [zf(z) for z in res[:3]],
        "swing_highs": [{"date": str(d.date()), "price": round(p, 2)}
                        for t, d, p in pivots if t == "high"][-3:],
        "swing_lows": [{"date": str(d.date()), "price": round(p, 2)}
                       for t, d, p in pivots if t == "low"][-3:],
        "levels": levels,
        "recent_candles": recent,
    }


def _level_breaks(df: pd.DataFrame, cur: float) -> dict:
    """
    Compute previous-day / 2-day / weekly high-low reference levels and whether
    TODAY (last bar) broke above, got rejected at, lost, or held each level.
    `df` is daily candles up to and including the as_of day (last row = today).
    """
    out = {}
    if len(df) < 2:
        return out
    today = df.iloc[-1]
    th, tl = float(today["high"]), float(today["low"])

    # Previous day
    pdh = float(df.iloc[-2]["high"]); pdl = float(df.iloc[-2]["low"])
    out["pdh"] = round(pdh, 2); out["pdl"] = round(pdl, 2)
    out["pdh_state"] = ("broke"   if cur > pdh else
                        "rejected" if th >= pdh and cur < pdh else "below")
    out["pdl_state"] = ("lost"    if cur < pdl else
                        "held"     if tl <= pdl and cur > pdl else "above")

    # 2-day (two sessions before today)
    if len(df) >= 3:
        w = df.iloc[-3:-1]
        d2h = float(w["high"].max()); d2l = float(w["low"].min())
        out["two_day_high"] = round(d2h, 2); out["two_day_low"] = round(d2l, 2)
        out["two_day_high_broken"] = cur > d2h
        out["two_day_low_broken"]  = cur < d2l

    # Weekly: resample to completed weeks, use the last FULLY completed week
    try:
        wk = df.resample("1W").agg({"high": "max", "low": "min"}).dropna()
        if len(wk) >= 2:
            pwh = float(wk.iloc[-2]["high"]); pwl = float(wk.iloc[-2]["low"])
            out["week_high"] = round(pwh, 2); out["week_low"] = round(pwl, 2)
            out["week_high_state"] = ("broke" if cur > pwh else
                                      "rejected" if th >= pwh and cur < pwh else "below")
            out["week_low_state"]  = ("lost"  if cur < pwl else
                                      "held"  if tl <= pwl and cur > pwl else "above")
    except Exception:
        pass
    return out


# ════════════════════════════════════════════════════════════════════
#  LLM DECISION (uses compact_prompt if available)
# ════════════════════════════════════════════════════════════════════
_PRE_SYS = ("You are a disciplined NSE equity trader. Protect capital first. "
            "R:R must be >= 1.5. Max risk 2% per trade. HOLD if setup unclear. "
            "Ground EVERY decision in the SPECIFIC features given (EMA stack, VWAP, "
            "PDH/PDL state, zones, VIX regime, fundamentals) — never invent data. "
            "Output JSON with keys: action|quantity|stop_loss|take_profit|risk_reward|"
            "reasoning|confidence|key_risk|features_used. "
            "'reasoning' = 2-3 short clauses each citing a named feature. "
            "'features_used' = list of the exact feature names you relied on, e.g. "
            "[\"ema_stack\",\"vwap\",\"pdh_state\",\"zone.1h.demand\",\"vix.regime\"].")
_POST_SYS = ("Review the completed trade and extract one concrete lesson. "
             "Output JSON: outcome|exit_reason_analysis|what_went_right|what_went_wrong|"
             "lesson_learned|strategy_adjustment")


def _build_user_prompt(symbol, date, tech, fund, capital, history):
    """Use compact_prompt.py if present, else a simple inline format."""
    try:
        from compact_prompt import build_compact_user
        return build_compact_user(symbol, date, tech, fund, capital, history, INITIAL_CAP)
    except Exception:
        ns, nr = tech.get("nearest_support"), tech.get("nearest_resistance")
        hist = "\n".join(f"  Ep{t['episode']:02d} {t['action']} {t['exit_reason']} "
                         f"P&L{t['pnl']:+.0f} | {t.get('lesson_learned','—')}"
                         for t in history[-4:]) or "  (none)"
        return (f"[{symbol} {date}]\nClose:{tech['close']} VWAP:{tech['vwap']}\n"
                f"EMA:{tech['ema_stack']} Trend:{tech['trend']}\n"
                f"Support:{ns} Resistance:{nr}\nHISTORY:\n{hist}\nCAP:{capital:.0f}")


def pre_trade_decision(symbol, date, tech, fund, capital, history) -> dict:
    user = _build_user_prompt(symbol, date, tech, fund, capital, history)
    raw = _llm_generate(_PRE_SYS, user, temp=0.2)
    dec = _parse_json(raw)
    if dec.get("action") in ("BUY", "SELL", "HOLD"):
        dec["_raw"] = raw
        return dec

    # Rule-based fallback
    ns, nr = tech.get("nearest_support"), tech.get("nearest_resistance")
    cur = tech["close"]
    max_risk = capital * RISK_PER_TRADE
    if tech["ema_stack"].startswith("bullish") and tech["above_vwap"] and ns:
        sl = round(ns["low"] * 0.998, 2)
        tp = round(nr["mid"] if nr else cur * 1.02, 2)
        rp = max(cur - sl, 0.01)
        qty = min(max(1, int(max_risk / rp)), max(1, int(capital // cur)))
        rr = round((tp - cur) / rp, 2)
        if rr >= 1.5:
            return {"action": "BUY", "quantity": qty, "stop_loss": sl, "take_profit": tp,
                    "risk_reward": rr, "confidence": 0.55,
                    "reasoning": f"Rule: bullish EMA stack, above VWAP, support {ns['mid']:.1f}",
                    "key_risk": "EMA reversal",
                    "features_used": ["ema_stack", "above_vwap", "nearest_support"]}
    return {"action": "HOLD", "quantity": 0, "stop_loss": 0, "take_profit": 0,
            "risk_reward": 0, "confidence": 0.4,
            "reasoning": "Rule: no clear setup (EMA stack/VWAP not aligned).", "key_risk": "—",
            "features_used": ["ema_stack", "above_vwap"]}


def post_trade_analysis(trade, forward_df) -> dict:
    outcome = "profit" if trade["pnl"] > 0 else "loss" if trade["pnl"] < 0 else "breakeven"
    fwd = forward_df[["high", "low", "close"]].round(2).to_string() if not forward_df.empty else "(none)"
    user = (f"[{trade['symbol']} Ep{trade['episode']}] {trade['action']} "
            f"{trade['quantity']}@{trade['entry_price']:.2f} SL:{trade['stop_loss']:.2f} "
            f"TP:{trade['take_profit']:.2f}\nEXIT:{trade['exit_price']:.2f} "
            f"[{trade['exit_reason']}] P&L:{trade['pnl']:+.0f}\n"
            f"Reasoning: {trade['pre_trade_reasoning']}\nForward candles:\n{fwd}")
    raw = _llm_generate(_POST_SYS, user, temp=0.1)
    res = _parse_json(raw)
    if "lesson_learned" not in res:
        res = {"outcome": outcome,
               "exit_reason_analysis": f"Exit via {trade['exit_reason']} at {trade['exit_price']:.2f}",
               "what_went_right": "setup met" if outcome == "profit" else "—",
               "what_went_wrong": "SL hit — review zones" if outcome == "loss" else "—",
               "lesson_learned": "Cut losses at SL" if outcome == "loss" else "Repeat demand-zone long",
               "strategy_adjustment": "Tighten entry criteria"}
    return res


# ════════════════════════════════════════════════════════════════════
#  SL/TP + helpers
# ════════════════════════════════════════════════════════════════════
def check_sl_tp(action, entry, sl, tp, fwd_df, max_hold=MAX_HOLD_DAYS):
    if action == "HOLD" or not sl or not tp or fwd_df.empty:
        return "HOLD", entry, None, 0
    for i, (idx, row) in enumerate(fwd_df.head(max_hold).iterrows()):
        lo, hi = float(row["low"]), float(row["high"])
        ds = str(idx.date())
        if action == "BUY":
            if lo <= sl: return "SL_HIT", round(sl, 2), ds, i + 1
            if hi >= tp: return "TP_HIT", round(tp, 2), ds, i + 1
        elif action == "SELL":
            if hi >= sl: return "SL_HIT", round(sl, 2), ds, i + 1
            if lo <= tp: return "TP_HIT", round(tp, 2), ds, i + 1
    last = fwd_df.head(max_hold).iloc[-1]
    return "TIME_EXIT", round(float(last["close"]), 2), str(fwd_df.head(max_hold).index[-1].date()), max_hold


def compute_pnl(action, entry, exit_p, qty):
    if action == "BUY":  return round((exit_p - entry) * qty, 2)
    if action == "SELL": return round((entry - exit_p) * qty, 2)
    return 0.0


def next_trading_day(date_str, offset=1):
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=offset)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


# ════════════════════════════════════════════════════════════════════
#  TRADING AGENT
# ════════════════════════════════════════════════════════════════════
class TradingAgent:
    def __init__(self, symbol: str, start_date: str, capital: float = INITIAL_CAP):
        self.symbol = symbol
        self.capital = capital
        self.peak_cap = capital
        self.history: list[dict] = []
        self.date = start_date
        self.episode = 0

    def alive(self) -> bool:
        return self.capital >= INITIAL_CAP * DEATH_THRESHOLD

    def mdd(self) -> float:
        caps = [INITIAL_CAP] + [t["capital_after"] for t in self.history]
        peak, mdd = caps[0], 0.0
        for c in caps:
            peak = max(peak, c); mdd = min(mdd, (c - peak) / peak * 100)
        return round(mdd, 2)

    def run_episode(self) -> dict:
        self.episode += 1
        date = self.date
        print(f"\n  EP {self.episode:02d} | {date} | Cap {self.capital:,.0f} | "
              f"{'ALIVE' if self.alive() else 'DANGER'}")

        candles = get_candles(self.symbol, date, LOOKBACK_DAYS)
        tech    = compute_technicals(candles)
        vc = vix_context(date)
        if vc:
            tech["vix_ctx"] = vc
        zc = compute_zones(self.symbol, date)
        if zc:
            tech["zones"] = zc
        fund    = get_fundamentals(self.symbol, date)
        tech["fund_completeness"] = _fund_completeness(fund, date)
        entry_p = tech["close"]

        dec = pre_trade_decision(self.symbol, date, tech, fund, self.capital, self.history)
        action = dec.get("action", "HOLD")
        qty = int(dec.get("quantity", 0))
        sl  = float(dec.get("stop_loss", 0))
        tp  = float(dec.get("take_profit", 0))
        rsn = dec.get("reasoning", "—")
        print(f"     -> {action} qty={qty} SL={sl:.2f} TP={tp:.2f}")

        fwd = get_forward_candles(self.symbol, date, MAX_HOLD_DAYS)
        exit_reason, exit_p, exit_date, bars = check_sl_tp(action, entry_p, sl, tp, fwd)
        pnl = compute_pnl(action, entry_p, exit_p, qty)
        cap_before = self.capital
        self.capital = round(self.capital + pnl, 2)
        self.peak_cap = max(self.peak_cap, self.capital)

        trade = {
            "episode": self.episode, "symbol": self.symbol, "decision_date": date,
            "entry_date": date, "action": action, "entry_price": round(entry_p, 2),
            "quantity": qty, "stop_loss": sl, "take_profit": tp,
            "exit_price": exit_p, "exit_date": exit_date, "exit_reason": exit_reason,
            "pnl": pnl, "capital_before": cap_before, "capital_after": self.capital,
            "pre_trade_reasoning": rsn, "post_trade_analysis": "—",
            "lesson_learned": "—", "what_went_wrong": "—",
        }
        mark = "PROFIT" if pnl > 0 else "LOSS" if pnl < 0 else "FLAT"
        print(f"     [{mark}] {exit_reason} @ {exit_p:.2f} | P&L {pnl:+.0f} | Cap {self.capital:,.0f}")

        if action != "HOLD":
            analysis = post_trade_analysis(trade, fwd)
            trade["post_trade_analysis"] = analysis.get("exit_reason_analysis", "—")
            trade["lesson_learned"] = analysis.get("lesson_learned", "—")
            trade["what_went_wrong"] = analysis.get("what_went_wrong", "—")

        tech_snap = {k: v for k, v in tech.items() if k != "_df"}
        with sqlite3.connect(DB_PATH) as con:
            con.execute("""INSERT INTO trade_log (
                episode,symbol,decision_date,action,entry_price,quantity,stop_loss,
                take_profit,exit_price,exit_date,exit_reason,pnl,capital_before,
                capital_after,pre_trade_reasoning,post_trade_analysis,lesson_learned,
                what_went_wrong,technicals_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                trade["episode"], trade["symbol"], trade["decision_date"], trade["action"],
                trade["entry_price"], trade["quantity"], trade["stop_loss"], trade["take_profit"],
                trade["exit_price"], trade["exit_date"], trade["exit_reason"], trade["pnl"],
                trade["capital_before"], trade["capital_after"], trade["pre_trade_reasoning"],
                trade["post_trade_analysis"], trade["lesson_learned"], trade["what_went_wrong"],
                json.dumps(tech_snap), datetime.now().isoformat()))

        self.history.append(trade)
        self.date = next_trading_day(date, 1)
        return trade
