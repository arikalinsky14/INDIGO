"""scripts/check_production_config.py: the tracked production records agree,
and a disagreement or a missing field is caught."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import check_production_config as chk  # noqa: E402


def test_tracked_production_records_agree():
    folders = sorted(p.parent for p in (REPO / "results" / "production").glob("*/PRODUCTION.json"))
    assert folders, "no tracked production record"
    for f in folders:
        assert chk.problems(f) == [], chk.problems(f)


def _write(tmp_path, prod, cfg):
    (tmp_path / "PRODUCTION.json").write_text(json.dumps(prod))
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


GOOD_P = {"lr": 3.104e-4, "batch_size": 256, "d_model": 352, "slot_encoder_layers": 6,
          "decoder_layers": 1, "n_heads": 11}
GOOD_C = {"learning_rate": 3.104e-4, "batch_size": 256, "d_model": 352,
          "slot_encoder_layers": 6, "decoder_layers": 1, "n_heads": 11}


def test_agreement_passes(tmp_path):
    assert chk.problems(_write(tmp_path, GOOD_P, GOOD_C)) == []


def test_the_nov18_defaults_fail(tmp_path):
    bad = {**GOOD_C, "learning_rate": 4.42e-5, "batch_size": 64}
    msgs = chk.problems(_write(tmp_path, GOOD_P, bad))
    assert len(msgs) == 2 and any("learning_rate" in m for m in msgs)


def test_each_architecture_field_is_checked(tmp_path):
    for k in ("d_model", "slot_encoder_layers", "decoder_layers", "n_heads"):
        msgs = chk.problems(_write(tmp_path, GOOD_P, {**GOOD_C, k: GOOD_C[k] + 1}))
        assert len(msgs) == 1 and k in msgs[0]


def test_missing_field_fails(tmp_path):
    p = {k: v for k, v in GOOD_P.items() if k != "batch_size"}
    assert any("missing" in m for m in chk.problems(_write(tmp_path, p, GOOD_C)))
