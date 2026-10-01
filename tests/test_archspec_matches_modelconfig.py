"""ArchSpec must stay a faithful stand-in for ModelConfig.

src/scaling/flops.py:ArchSpec exists so the scaling analysis can count
parameters without importing torch. It copies ModelConfig's defaults by hand,
which is a drift risk: if someone changes a default in src/model.py and not
here, every parameter count in the ladder shifts and the sweep silently plans
a different grid. Nothing would look wrong.

So: compare every field flops.py reads, and check that the two give identical
parameter and FLOP counts across the ladder.
"""
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Fields src/scaling/flops.py reads off a config.
READS = ("batch_size", "d_model", "decoder_layers", "encoder_hidden",
         "encoder_out", "feature_mode", "head_mode", "n_layers",
         "slot_encoder_layers", "vocab_size")

LADDER = ((32, 1), (40, 1), (64, 2), (128, 2), (192, 4), (288, 5), (416, 7))


def test_defaults_match():
    from src.model import ModelConfig
    from src.scaling.flops import ArchSpec
    mc = {f.name: f.default for f in fields(ModelConfig)}
    asp = {f.name: f.default for f in fields(ArchSpec)}
    for name in READS:
        assert name in asp, f"ArchSpec is missing {name!r}, which flops.py reads"
        assert asp[name] == mc[name], (
            f"default for {name!r} drifted: ModelConfig has {mc[name]!r}, "
            f"ArchSpec has {asp[name]!r}")


def test_counts_match_across_the_ladder():
    from src.model import ModelConfig
    from src.scaling.flops import (ArchSpec, n_params,
                                   forward_flops_per_example)
    from src.scaling.configs import n_heads_for
    for d_model, se in LADDER:
        kw = dict(head_mode="cross_attn", d_model=d_model,
                  n_heads=n_heads_for(d_model), slot_encoder_layers=se,
                  decoder_layers=1)
        a, m = ArchSpec(**kw), ModelConfig(**kw)
        assert n_params(a) == n_params(m), (
            f"d{d_model}/se{se}: ArchSpec says {n_params(a):,} parameters, "
            f"ModelConfig says {n_params(m):,}")
        assert forward_flops_per_example(a) == forward_flops_per_example(m), (
            f"d{d_model}/se{se}: forward FLOPs differ")


if __name__ == "__main__":
    test_defaults_match()
    test_counts_match_across_the_ladder()
    print("PASS")
