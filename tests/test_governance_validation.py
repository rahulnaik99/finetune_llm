"""Governance artifacts (model card + datasheet) and data validation."""
import common  # noqa: F401 — path bootstrap


def test_model_card_and_datasheet_generate():
    from modules.governance.service import generate_model_card, generate_datasheet
    mc = generate_model_card()
    ds = generate_datasheet()
    from pathlib import Path
    mc_txt = Path(mc).read_text()
    ds_txt = Path(ds).read_text()
    assert "Model Card" in mc_txt and "Intended use" in mc_txt and "Limitations" in mc_txt
    assert "Datasheet" in ds_txt and "Composition" in ds_txt


def test_validation_accepts_good_csv_and_flags_bad(tmp_path):
    from common.validation import validate_price_csv
    good = tmp_path / "good.csv"
    good.write_text("date,open,high,low,close,volume\n"
                    "2024-01-01,10,11,9,10.5,100\n"
                    "2024-01-02,10.5,12,10,11.5,120\n")
    r = validate_price_csv(str(good))
    assert r["ok"] is True and r["rows"] == 2

    bad = tmp_path / "bad.csv"
    bad.write_text("date,open,high,low,close\n2024-01-01,10,5,9,10\n")  # high < low
    rb = validate_price_csv(str(bad))
    assert rb["ok"] is False and rb["errors"]


def test_validation_missing_file():
    from common.validation import validate_price_csv
    r = validate_price_csv("/nope/does_not_exist.csv")
    assert r["ok"] is False
