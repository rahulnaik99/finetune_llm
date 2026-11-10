"""
common/validation.py
────────────────────
Lightweight data-quality gate for the input CSVs (prices, India VIX) — the
industry practice of *validating data before it enters the pipeline* (cf. Great
Expectations / pandera / TFDV).

Uses **pandera** when installed for a declarative schema; otherwise falls back to
a dependency-free manual check so validation always runs (even in CI). Returns a
report dict rather than raising, so callers decide whether a failure is fatal.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

OHLC = ["open", "high", "low", "close"]


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _manual_checks(df: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    for col in OHLC:
        if col not in df.columns:
            errors.append(f"missing column '{col}'")
    if errors:
        return errors
    num = df[OHLC].apply(pd.to_numeric, errors="coerce")
    if num.isna().any().any():
        errors.append("non-numeric or NaN values in OHLC")
    if (num[["open", "high", "low", "close"]] <= 0).any().any():
        errors.append("non-positive prices present")
    # high must be the max, low the min of the bar
    bad_hl = (num["high"] < num["low"]).sum()
    if bad_hl:
        errors.append(f"{int(bad_hl)} rows where high < low")
    return errors


def validate_price_csv(path: str) -> dict:
    """Validate an OHLCV (or VIX) CSV. Returns {ok, rows, errors, engine}."""
    p = Path(path)
    if not p.exists():
        return {"ok": False, "rows": 0, "errors": [f"file not found: {path}"], "engine": "none"}
    df = _load(path)
    try:
        import pandera as pa
        from pandera import Column, Check, DataFrameSchema

        schema = DataFrameSchema(
            {c: Column(float, Check.gt(0), coerce=True, nullable=False) for c in OHLC},
            strict=False,
        )
        schema.validate(df[[c for c in OHLC if c in df.columns]], lazy=True)
        extra = _manual_checks(df)   # high>=low isn't expressible column-wise
        return {"ok": not extra, "rows": len(df), "errors": extra, "engine": "pandera"}
    except ImportError:
        errors = _manual_checks(df)
        return {"ok": not errors, "rows": len(df), "errors": errors, "engine": "manual"}
    except Exception as exc:        # pandera SchemaErrors etc.
        return {"ok": False, "rows": len(df), "errors": [str(exc)[:300]], "engine": "pandera"}


def validate_inputs(prices_csv: str, vix_csv: Optional[str] = None) -> dict:
    report = {"prices": validate_price_csv(prices_csv)}
    if vix_csv:
        report["vix"] = validate_price_csv(vix_csv)
    report["ok"] = all(r["ok"] for r in report.values() if isinstance(r, dict))
    return report
