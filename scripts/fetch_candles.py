"""
scripts/fetch_candles.py
────────────────────────
Fetches historical OHLCV candles from yfinance and stores as Parquet shards.
Referenced by dvc.yaml as Stage 1 of the pipeline.

Usage:
    python scripts/fetch_candles.py \
        --symbols CANBK SBIN HDFCBANK \
        --start 2021-01-01 \
        --end   2024-10-10 \
        --out   data/candles
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def fetch_symbol(symbol: str, start: str, end: str, out_dir: str, tfs: list[str]) -> dict:
    from candle_store import CandleStore
    import yfinance as yf
    import pandas as pd

    store = CandleStore(out_dir)
    stats = {"symbol": symbol}

    TF_MAP = {"1m": "1m", "15m": "15m", "1h": "1h", "1d": "1d"}
    YF_PERIOD = {"1m": None, "15m": None, "1h": None, "1d": None}
    YF_INTERVAL = {"1m": "1m", "15m": "15m", "1h": "1h", "1d": "1d"}

    # yfinance max lookback per interval
    MAX_DAYS = {"1m": 7, "15m": 60, "1h": 730, "1d": 3650}

    from datetime import datetime, timedelta

    for tf in tfs:
        max_d = MAX_DAYS.get(tf, 730)
        # Chunk into max_d-day windows (yfinance limit for intraday)
        s_dt = datetime.fromisoformat(start)
        e_dt = datetime.fromisoformat(end)
        total = 0

        current = s_dt
        while current < e_dt:
            chunk_end = min(current + timedelta(days=max_d), e_dt)
            try:
                df = yf.download(
                    f"{symbol}.NS",
                    start=current.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval=YF_INTERVAL[tf],
                    progress=False, auto_adjust=True,
                )
                if df.empty:
                    current = chunk_end
                    continue
                # Flatten MultiIndex (yfinance >=0.2)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                df = df.reset_index().rename(columns={"Datetime": "ts", "Date": "ts"})
                if "ts" not in df.columns and "index" in df.columns:
                    df = df.rename(columns={"index": "ts"})
                n = store.save_candles(symbol, tf, df)
                total += n
                print(f"  {symbol} {tf} {current.date()} → {chunk_end.date()}: {n} bars")
            except Exception as e:
                print(f"  {symbol} {tf} {current.date()} ERROR: {e}")
            current = chunk_end

        stats[tf] = total
        print(f"  {symbol} {tf}: {total} total bars saved")

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start",   default="2021-01-01")
    p.add_argument("--end",     default="2024-10-10")
    p.add_argument("--out",     default="data/candles")
    p.add_argument("--tfs",     nargs="+", default=["1d", "1h"],
                   help="Timeframes to fetch (1m needs short windows, slow)")
    args = p.parse_args()

    print(f"Fetching {args.tfs} candles for {len(args.symbols)} symbols")
    print(f"Date range: {args.start} → {args.end}")
    print(f"Output: {args.out}\n")

    all_stats = []
    for sym in args.symbols:
        print(f"\n── {sym} ──────────────────────────────────────────────")
        stats = fetch_symbol(sym, args.start, args.end, args.out, args.tfs)
        all_stats.append(stats)

    print("\n── Summary ───────────────────────────────────────────────")
    for s in all_stats:
        sym = s.pop("symbol")
        print(f"  {sym}: " + "  ".join(f"{tf}={v}" for tf, v in s.items()))
    print("\nDone.")


if __name__ == "__main__":
    main()
