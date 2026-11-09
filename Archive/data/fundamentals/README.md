# Fundamentals from official SEBI XBRL filings

Put one XBRL file per quarter per symbol here, in a per-symbol folder:

```
data/fundamentals/
  CANBK/
    CANBK_2024Q1.xml
    CANBK_2024Q2.xml
    CANBK_2026Q2.xml
    ...
  SBIN/
    SBIN_2024Q1.xml
    ...
```

Filenames don't matter — the parser reads the dates and nature from inside each
file. Download the "Integrated Filing - Financials (XBRL)" attachments from the
NSE/BSE filing portal (one per quarter).

## Which files to keep

For each quarter you'll often have several rows on the portal:
- **Consolidated vs Standalone** — keep whichever you train on (default: Consolidated).
  You can keep both; the builder filters by `--nature`.
- **Original vs Revision** — keep both if you like; the builder automatically
  picks the latest / most-complete one per quarter (revisions win).

## Build the LLM-ready JSON (QoQ / YoY / TTM / ratios pre-computed)

```bash
python fundamentals_xbrl.py --build CANBK --dir data/fundamentals --nature Consolidated
# writes data/fundamentals/CANBK/fundamentals.json
```

Every quarter in that JSON already contains:
- `pat_cr`, `pat_qoq_pct`, `pat_yoy_pct`
- `net_interest_income_cr`, `nii_yoy_pct`
- `total_income_cr`, `income_yoy_pct`, `operating_profit_cr`, `opprofit_yoy_pct`
- `eps`, `eps_ttm`
- `gnpa_pct`, `gnpa_qoq_bps`, `gnpa_yoy_bps`, `nnpa_pct`, `provision_coverage_pct`
- `cet1_pct`, `roa_pct`
- `deposits_cr`, `advances_cr`, `cd_ratio_pct`, `deposit_yoy_pct`, `advance_yoy_pct`
- `summary` — a one-line, LLM-ready sentence with all the deltas

The simulation reads this automatically when `.env` has:
```ini
FUNDAMENTALS_SOURCE=xbrl
XBRL_DIR=data/fundamentals
```
It serves the latest quarter whose **result date ≤ the simulation date** (no look-ahead),
and if the JSON is missing it is built on first use.

## Notes on units
- Money facts are full rupees in the XBRL; the JSON converts to **crore** (`÷1e7`).
- Ratios like GNPA are fractions in the XBRL (`0.0133`); the JSON shows **percent** (`1.33`).
- Changes in ratios are reported in **basis points** (100 bps = 1%).
