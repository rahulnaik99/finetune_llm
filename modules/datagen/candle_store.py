"""
candle_store.py
───────────────
Production-grade candle data store using Polars + DuckDB.
Replaces raw pandas reads for 4-year 1-minute candle data.

Key APIs:
  - save_candles()     write OHLCV data to monthly Parquet shards
  - load_candles()     fast date-range query via DuckDB (no full-file scan)
  - load_multi_tf()    load daily/hourly/15m/1m from the same Parquet store
  - compute_features() EMA/VWAP/ATR/momentum using Polars (no pandas needed)

Why Polars + DuckDB over pandas:
  - Polars: lazy evaluation, multi-threaded, 5-10× faster on large DataFrames
  - DuckDB: SQL on Parquet without loading entire files — critical for 4yr 1-min
  - Both are widely asked about in 2025 data/ML engineering interviews

Usage:
    store = CandleStore("data/candles")
    df = store.load_candles("CANBK", "1m", "2024-01-01", "2024-10-10")
    features = store.compute_features(df)
"""
"""REFERENCE IMPLEMENTATION — NOT wired into the live pipeline.

This Polars+DuckDB Parquet candle store is a production-grade data layer for
1-minute, multi-year data (SQL-on-Parquet, no full-file scans). The simulation
currently reads candles via pandas/CSV/yfinance (see simulation.get_candles).
To adopt it, route get_candles()/compute_zones() through CandleStore. Kept as a
reference so the capability is documented without being a dead dependency.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


class CandleStore:
    """Manages Parquet-backed OHLCV data for multiple symbols and timeframes."""

    def __init__(self, base_dir: str = "data/candles"):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _shard_path(self, symbol: str, tf: str, year: int, month: int) -> Path:
        """Monthly shards: data/candles/CANBK/1m/2024-01.parquet"""
        return self.base / symbol / tf / f"{year:04d}-{month:02d}.parquet"

    # ── Write ─────────────────────────────────────────────────────────────────
    def save_candles(self, symbol: str, tf: str, df_in) -> int:
        """
        Save OHLCV DataFrame to Parquet shards.
        Accepts both pandas and Polars DataFrames.
        Returns total rows written.
        """
        import polars as pl

        # Normalise to Polars
        if hasattr(df_in, "to_pandas"):          # already Polars
            df = df_in
        else:                                     # pandas → Polars
            df = pl.from_pandas(df_in)

        # Ensure datetime column
        if "ts" not in df.columns and "time" in df.columns:
            df = df.rename({"time": "ts"})
        df = df.with_columns(pl.col("ts").cast(pl.Datetime))

        # Write one Parquet per month
        total = 0
        for (year, month), group in df.group_by([
            pl.col("ts").dt.year().alias("y"),
            pl.col("ts").dt.month().alias("m"),
        ]):
            path = self._shard_path(symbol, tf, year, month)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Merge with existing shard if present
            if path.exists():
                existing = pl.read_parquet(path)
                group = pl.concat([existing, group.drop(["y","m"])]).unique("ts").sort("ts")
            else:
                group = group.drop(["y","m"]).sort("ts")
            group.write_parquet(path, compression="zstd")
            total += len(group)
        return total

    # ── Read via DuckDB (never loads full files) ──────────────────────────────
    def load_candles(
        self,
        symbol: str,
        tf: str,
        start: str,
        end: str,
    ):
        """
        Load candles for a date range using DuckDB predicate pushdown.
        Much faster than loading all Parquet files and filtering in Python.
        """
        try:
            import duckdb, polars as pl
        except ImportError:
            return self._load_pandas_fallback(symbol, tf, start, end)

        shard_dir = self.base / symbol / tf
        if not shard_dir.exists():
            raise FileNotFoundError(f"No candle data at {shard_dir}")

        # DuckDB reads only the row groups it needs — no full scan
        result = duckdb.sql(f"""
            SELECT ts, open, high, low, close, volume
            FROM read_parquet('{shard_dir}/*.parquet')
            WHERE ts >= TIMESTAMP '{start}'
              AND ts <= TIMESTAMP '{end}'
            ORDER BY ts
        """).pl()   # → Polars DataFrame directly

        return result

    def _load_pandas_fallback(self, symbol, tf, start, end):
        """Fallback when DuckDB is not installed (slower, loads full files)."""
        import pandas as pd
        shard_dir = self.base / symbol / tf
        parts = []
        for f in sorted(shard_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            df = df[(df["ts"] >= start) & (df["ts"] <= end)]
            if not df.empty:
                parts.append(df)
        return pd.concat(parts).reset_index(drop=True) if parts else pd.DataFrame()

    # ── Multi-timeframe loader ────────────────────────────────────────────────
    def load_multi_tf(
        self,
        symbol: str,
        as_of: str,
        lookbacks: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        import polars as pl  # ensure pl is in local scope for filter()
        """
        Load multiple timeframes as a dict of Polars DataFrames.
        lookbacks = {"1m": 60, "15m": 32, "1h": 48, "1d": 60}
        """
        if lookbacks is None:
            lookbacks = {"1m": 60, "15m": 32, "1h": 48, "1d": 60}

        result = {}
        end_dt = datetime.fromisoformat(as_of)

        for tf, n_bars in lookbacks.items():
            # Estimate how many calendar days we need
            mins_per_bar = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "1d": 1440}.get(tf, 1)
            buffer_days  = max(5, int(n_bars * mins_per_bar / 390) + 10)
            start_dt     = end_dt - timedelta(days=buffer_days)

            try:
                df = self.load_candles(symbol, tf,
                                       start_dt.strftime("%Y-%m-%d"),
                                       as_of)
                # Polars: take last n_bars strictly before/at as_of
                if hasattr(df, "filter"):        # Polars
                    df = df.filter(pl.col("ts") <= end_dt).tail(n_bars)
                else:                            # pandas fallback
                    df = df[df["ts"] <= as_of].tail(n_bars)
                result[tf] = df
            except FileNotFoundError:
                result[tf] = None

        return result

    # ── Feature computation (pure Polars, no pandas) ─────────────────────────
    def compute_features(self, df) -> dict[str, Any]:
        """
        Compute EMA 50/100/200, VWAP, ATR, momentum from a Polars DataFrame.
        Returns a features dict matching the simulation's technicals schema.
        """
        import polars as pl

        if df is None or (hasattr(df, "__len__") and len(df) == 0):
            return {}

        # Convert pandas → Polars if needed
        if not hasattr(df, "lazy"):
            import polars as pl
            df = pl.from_pandas(df)

        close  = df["close"].to_numpy()
        high   = df["high"].to_numpy()
        low    = df["low"].to_numpy()
        volume = df["volume"].to_numpy()

        def ema_np(arr, span):
            k = 2 / (span + 1)
            out = np.zeros_like(arr, dtype=float)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = arr[i] * k + out[i-1] * (1 - k)
            return out

        e50  = ema_np(close, 50)
        e100 = ema_np(close, 100)
        e200 = ema_np(close, 200)
        tp   = (high + low + close) / 3
        vwap = np.cumsum(tp * volume) / np.cumsum(volume)
        atr  = np.mean(high[-14:] - low[-14:]) if len(close) >= 14 else high[-1] - low[-1]
        mom3d= float((close[-1] / close[-4] - 1) * 100) if len(close) >= 4 else 0.0

        cur = float(close[-1])
        stack = ("bullish (50>100>200)" if e50[-1] > e100[-1] > e200[-1] else
                 "bearish (50<100<200)" if e50[-1] < e100[-1] < e200[-1] else "mixed")

        return {
            "close":   round(cur, 2),
            "ema_50":  round(float(e50[-1]), 2),
            "ema_100": round(float(e100[-1]), 2),
            "ema_200": round(float(e200[-1]), 2),
            "vwap":    round(float(vwap[-1]), 2),
            "atr14":   round(float(atr), 2),
            "momentum_3d_pct": round(mom3d, 2),
            "ema_stack": stack,
            "above_vwap":   cur > float(vwap[-1]),
            "above_ema200": cur > float(e200[-1]),
        }


# ── DuckDB analytics: cross-symbol market movers (production alternative to yfinance batch) ──
def market_movers_duckdb(candle_dir: str, symbols: list[str], as_of: str, lookback_days: int = 30):
    """
    Compute daily movers across all symbols using DuckDB on local Parquet.
    Much faster than yfinance batch download once data is cached locally.
    Demonstrates the DuckDB pattern that comes up in data engineering interviews.
    """
    try:
        import duckdb
    except ImportError:
        raise ImportError("pip install duckdb")

    base = Path(candle_dir)
    results = []

    for sym in symbols:
        daily_dir = base / sym / "1d"
        if not daily_dir.exists():
            continue
        try:
            rows = duckdb.sql(f"""
                WITH recent AS (
                    SELECT ts::DATE as date, close, volume
                    FROM read_parquet('{daily_dir}/*.parquet')
                    WHERE ts::DATE >= (DATE '{as_of}' - INTERVAL {lookback_days} DAYS)
                      AND ts::DATE <= DATE '{as_of}'
                    ORDER BY ts
                )
                SELECT
                    '{sym}'                                              as symbol,
                    last(close)                                          as close,
                    last(close) / nth_value(close, 2 OVER (ORDER BY date DESC)) - 1 as chg_1d,
                    last(close) / first(close) - 1                      as chg_lookback,
                    avg(volume)                                          as avg_vol,
                    last(volume) / avg(volume)                          as vol_ratio
                FROM recent
            """).fetchone()
            if rows:
                results.append(dict(zip(
                    ["symbol","close","chg_1d","chg_lookback","avg_vol","vol_ratio"], rows
                )))
        except Exception:
            continue

    return sorted(results, key=lambda r: r.get("chg_1d", 0) or 0, reverse=True)


if __name__ == "__main__":
    # Quick demo — save synthetic data and load it back
    import polars as pl
    import numpy as np
    from datetime import datetime, timedelta

    print("CandleStore demo — synthetic CANBK 1-min data")
    store = CandleStore("/tmp/candle_demo")

    # Generate 5 days × 390 min = 1,950 rows
    base = datetime(2024, 10, 7, 9, 15)
    ts = [base + timedelta(minutes=i) for i in range(1950)]
    close = 119 + np.cumsum(np.random.randn(1950) * 0.05)
    df = pl.DataFrame({
        "ts":     ts,
        "open":   close - np.random.rand(1950) * 0.3,
        "high":   close + np.random.rand(1950) * 0.5,
        "low":    close - np.random.rand(1950) * 0.5,
        "close":  close,
        "volume": np.random.randint(10000, 50000, 1950),
    })
    n = store.save_candles("CANBK", "1m", df)
    print(f"  Saved {n} rows")

    loaded = store.load_candles("CANBK", "1m", "2024-10-07", "2024-10-11")
    print(f"  Loaded {len(loaded)} rows via DuckDB")

    features = store.compute_features(loaded)
    print(f"  Features: close={features['close']} ema50={features['ema_50']} stack={features['ema_stack']}")
    print("Demo complete")
