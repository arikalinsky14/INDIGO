"""fit_lr_law --diverged-brackets: a divergence above closes a top-edge optimum."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_lr_law import cliff_bracketed, tune_per_config  # noqa: E402


def cells(lrs, vals, diverged=(), n=1_000_000, d=5_000_000):
    out = [{"n_params": n, "train_examples": d, "lr": lr, "batch_size": 256.0,
            "beta2": 0.999, "d_model": 128, "slot_encoder_layers": 3,
            "value": v, "diverged": False} for lr, v in zip(lrs, vals)]
    out += [{"n_params": n, "train_examples": d, "lr": lr, "batch_size": 256.0,
             "beta2": 0.999, "d_model": 128, "slot_encoder_layers": 3,
             "value": float("inf"), "diverged": True} for lr in diverged]
    return out


LRS = [1e-4, 2e-4, 4e-4, 8e-4]
FALLING = [12.0, 11.0, 10.5, 10.0]          # best at the top stable rate


def test_rule():
    assert cliff_bracketed(LRS, 8e-4, [1.6e-3])
    assert not cliff_bracketed(LRS, 8e-4, [])              # nothing diverged
    assert not cliff_bracketed(LRS, 1e-4, [1.6e-3])        # optimum at the bottom
    assert not cliff_bracketed(LRS[:2], 2e-4, [4e-4])      # two stable rates


def test_off_by_default_and_on_with_flag():
    c = cells(LRS, FALLING, diverged=[1.6e-3])
    strict = tune_per_config(c)[0]
    assert not strict["usable"] and strict["bracket"] == "open"
    loose = tune_per_config(c, diverged_brackets=True)[0]
    assert loose["usable"] and loose["bracket"] == "cliff"
    assert loose["lr_star"] <= 8e-4 and loose["n_diverged"] == 1


def test_bottom_edge_stays_open():
    c = cells(LRS, FALLING[::-1], diverged=[1.6e-3])
    assert not tune_per_config(c, diverged_brackets=True)[0]["usable"]


def test_interior_untouched():
    c = cells(LRS + [1.6e-3], [12.0, 11.0, 10.0, 11.0, 12.0], diverged=[3.2e-3])
    r = tune_per_config(c, diverged_brackets=True)[0]
    assert r["usable"] and r["bracket"] == "interior"
