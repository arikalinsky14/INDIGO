"""
Parameter and FLOP accounting for INDIGO models
===============================================

Phase 0.1 of the compute-optimal scaling study. Everything downstream
(`src/scaling/configs.py`, `scripts/scaling_sweep.py`,
`scripts/fit_scaling.py`) derives its compute axis from this module, so it
is written to be exact and auditable rather than clever.

Why this module exists (do not substitute `C = 6ND`)
----------------------------------------------------
The standard language-model heuristic `C_train = 6 * N * D_tokens` assumes
every parameter is used exactly once per token. That assumption is badly
violated by `FlexMaterialCrossAttn`:

  * `slot_encoder` runs over M_MAX = 32 pool slots per example, not over
    the 11-position decoder sequence.
  * `thickness_head` is applied per (position, slot), i.e. 11 * 32 = 352
    times per example. Its 2.2M parameters therefore contribute ~30-80%
    of all forward FLOPs depending on depth.

Measured consequence (validated against `torch.utils.flop_counter`):

    d_model  sel  N          fwd FLOPs/example   ratio to 2N
    1024     4    69.57M     5.235e9             37.6x
    512      4    17.51M     1.330e9             38.0x
    256      2     2.87M     2.436e8             42.5x
    128      2     0.77M     6.774e7             44.2x

So the per-token heuristic understates training compute by roughly an
order of magnitude, and the error is *not* a constant: the ratio drifts
from ~33 to ~44 across the sweep grid because the head's second layer
scales as `d_model * NUM_THICKNESSES` (linear in d) while the transformer
body scales as `d_model**2`. A single fudge factor cannot absorb that, which
is exactly why this is an analytic calculator and not a constant.

This is the concrete form of "last-layer FLOP accounting" (correction #1 of
Porian et al. 2024, arXiv:2406.19146). Note it is a much larger correction
here than the paper's: their concern is a vocabulary projection that matters
only at small N, whereas INDIGO's factorized pointer head is a leading-order
term at *every* scale.

Caveat on `torch.utils.flop_counter` (read before cross-checking)
-----------------------------------------------------------------
In `.eval()` mode under `torch.no_grad()`, `FlopCounterMode` reports ZERO
FLOPs for `slot_encoder`: `nn.TransformerEncoder` takes PyTorch's fused fast
path, which the counter does not instrument. On the production config that
hides 72% of the parameters and undercounts the forward pass by 2.7x
(5.235e9 -> 1.922e9 FLOPs/example) with no warning of any kind.

Both conditions are required, and the trap is that their conjunction is
exactly how one would naturally profile inference:

    training=True,  grad      ->  slot_encoder counted   5.235e9  (correct)
    training=True,  no_grad   ->  slot_encoder counted   5.235e9  (correct)
    training=False, grad      ->  slot_encoder counted   5.235e9  (correct)
    training=False, no_grad   ->  slot_encoder = 0       1.922e9  (WRONG)

`verify_against_torch()` therefore measures with gradients enabled. A
separate ~0.4% discrepancy depends on whether `dropout` was nonzero at
construction time (it changes which sub-ops get fused); that one sits within
the noise floor of the omitted LayerNorm/bias/activation terms and is not
worth chasing.

`test_no_grad_undercount_regression()` locks this behaviour down so that a
future torch upgrade which fixes (or worsens) it does not silently change
what we believe about the compute axis.

Conventions
-----------
* FLOPs follow the `FlopCounterMode` convention: one matmul of shape
  (m, k) @ (k, n) costs `2 * m * n * k`. Bias adds, LayerNorms, activations
  and softmaxes are omitted; they are O(d) rather than O(d^2) and account
  for the residual 0.4-1.8% disagreement with the counter.
* `D` is counted in **examples**, not tokens. One example is one
  (Lab target, material pool) pair. Token counts are reported separately
  via `tokens_per_example()` because a "token" here is a deposited layer,
  and the mean structure length (~4.5, set by `LAYER_LAMBDA`) is a dataset
  property rather than a model property.
* Training cost is `(1 + backward_multiplier) * forward`, with
  `backward_multiplier = 2.0` as usual. Activation recomputation would
  raise this; INDIGO does not use it.
* Parameter counts here are EXACT and unit-tested against `build_model`.
  FLOP counts are analytic approximations accurate to ~2%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.material_features import feature_dim
from src.materials_vocab import (
    M_MAX,
    MAX_LAYERS,
    NUM_THICKNESSES,
)

# Decoder sequence length: start token + one slot per depositable layer.
# Mirrors `FlexMaterialCrossAttn.SEQ_LEN`; duplicated here so this module
# imports no torch.
SEQ_LEN: int = MAX_LAYERS + 1

# Mean layers per example implied by the generator defaults in
# `slurms/generate_data.sh` (LAYER_LAMBDA=4.5, truncated to [2, 10]).
# Only used to convert examples <-> tokens for reporting.
DEFAULT_MEAN_LAYERS: float = 4.5


# ============================================================================
# Results
# ============================================================================


@dataclass
class Counts:
    """A component-wise breakdown plus its total."""

    by_component: Dict[str, int | float] = field(default_factory=dict)

    @property
    def total(self) -> int | float:
        return sum(self.by_component.values())

    def fraction(self, component: str) -> float:
        t = self.total
        return self.by_component.get(component, 0) / t if t else 0.0

    def table(self, unit: float = 1.0, unit_name: str = "") -> str:
        lines = []
        t = self.total
        for name, v in sorted(
            self.by_component.items(), key=lambda kv: -kv[1]
        ):
            pct = 100.0 * v / t if t else 0.0
            lines.append(f"  {name:<22} {v / unit:>14,.3f}{unit_name} {pct:>6.2f}%")
        lines.append(f"  {'TOTAL':<22} {t / unit:>14,.3f}{unit_name} {100.0:>6.2f}%")
        return "\n".join(lines)


# ============================================================================
# Architecture resolution
# ============================================================================


def slot_encoder_depth(config) -> int:
    """Effective slot-encoder depth.

    `src/model.py` resolves this as `slot_encoder_layers or n_layers`, so
    `n_layers` is a NO-OP whenever `slot_encoder_layers` is nonzero. The
    scaling study varies `slot_encoder_layers` and leaves `n_layers` at its
    default; this helper keeps that subtlety in exactly one place.
    """
    return config.slot_encoder_layers or config.n_layers


def _transformer_encoder_layer_params(d: int) -> int:
    """`nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4d)`.

    Head count does not appear: `in_proj_weight` is [3d, d] regardless of
    how many heads that is split into, so parameters (and FLOPs) are exactly
    invariant to `n_heads`. Verified empirically.
    """
    return (
        3 * d * d + 3 * d        # self_attn.in_proj (qkv)
        + d * d + d              # self_attn.out_proj
        + d * 4 * d + 4 * d      # linear1
        + 4 * d * d + d          # linear2
        + 2 * d + 2 * d          # norm1, norm2
    )                            # == 12d^2 + 13d


def _transformer_decoder_layer_params(d: int) -> int:
    """`nn.TransformerDecoderLayer(d, nhead, dim_feedforward=4d)`."""
    return (
        3 * d * d + 3 * d + d * d + d    # self_attn
        + 3 * d * d + 3 * d + d * d + d  # multihead_attn (cross)
        + d * 4 * d + 4 * d              # linear1
        + 4 * d * d + d                  # linear2
        + 3 * (2 * d)                    # norm1, norm2, norm3
    )                                    # == 16d^2 + 19d


# ============================================================================
# Parameters
# ============================================================================


def param_counts(config) -> Counts:
    """Exact parameter breakdown, matching `build_model(config)` bit for bit."""
    if config.head_mode == "cross_attn":
        return _param_counts_cross_attn(config)
    if config.head_mode == "mlp":
        return _param_counts_mlp(config)
    raise ValueError(f"Unknown head_mode {config.head_mode!r}")


def _material_encoder_params(config) -> int:
    e_in = feature_dim(config.feature_mode)
    eh, eo = config.encoder_hidden, config.encoder_out
    return e_in * eh + eh + eh * eo + eo


def _param_counts_cross_attn(config) -> Counts:
    d = config.d_model
    eo = config.encoder_out
    sel = slot_encoder_depth(config)
    dec = config.decoder_layers
    return Counts({
        "material_encoder": _material_encoder_params(config),
        "slot_proj": eo * d + d,
        "slot_encoder": sel * _transformer_encoder_layer_params(d),
        "past_proj": eo * d + d,
        "layer_pos_emb": SEQ_LEN * d,
        "lab_proj": 4 * d + d,
        "start_token": d,
        "decoder": dec * _transformer_decoder_layer_params(d),
        "thickness_head": (2 * d * d + d) + (d * NUM_THICKNESSES + NUM_THICKNESSES),
        "eos_head": d + 1,
    })


def _param_counts_mlp(config) -> Counts:
    d = config.d_model
    input_dim = 3 + M_MAX * config.encoder_out + M_MAX * MAX_LAYERS + 1
    hidden = (config.n_layers - 1) * (d * d + d)
    return Counts({
        "material_encoder": _material_encoder_params(config),
        "backbone_in": input_dim * d + d,
        "backbone_hidden": hidden,
        "output_head": d * config.vocab_size + config.vocab_size,
    })


def n_params(config) -> int:
    """Total trainable parameters."""
    return int(param_counts(config).total)


# ============================================================================
# Forward FLOPs (per example)
# ============================================================================


def forward_flop_counts(config, packed: Optional[bool] = None,
                        mean_layers: float = DEFAULT_MEAN_LAYERS) -> Counts:
    """Forward FLOPs for ONE example, broken down by component.

    Parameters
    ----------
    packed
        Whether teacher forcing is packed (one forward covering all
        `SEQ_LEN` positions). `None` resolves to the training default from
        `scripts/training.py:537`: on for `cross_attn`, off for `mlp`.
        Unpacked costs one forward per decoding step, which is why
        `slurms/training.sh` notes cross_attn "benefits ~5x" from packing.
    mean_layers
        Only used when `packed=False`, to count decoding steps per example.
    """
    if packed is None:
        packed = config.head_mode == "cross_attn"

    if config.head_mode == "cross_attn":
        counts = _forward_flop_counts_cross_attn(config)
    elif config.head_mode == "mlp":
        counts = _forward_flop_counts_mlp(config)
    else:
        raise ValueError(f"Unknown head_mode {config.head_mode!r}")

    if not packed:
        # One forward per decoding step (+1 for the EOS step).
        steps = mean_layers + 1.0
        counts = Counts({k: v * steps for k, v in counts.by_component.items()})
    return counts


def _forward_flop_counts_cross_attn(config) -> Counts:
    d = config.d_model
    e_in = feature_dim(config.feature_mode)
    eh, eo = config.encoder_hidden, config.encoder_out
    sel = slot_encoder_depth(config)
    dec = config.decoder_layers
    M, S, L, NT = M_MAX, SEQ_LEN, MAX_LAYERS, NUM_THICKNESSES

    # Slot self-attention: qkv + out_proj = 8Md^2, scores + AV = 4M^2 d,
    # FFN (d -> 4d -> d) = 16Md^2.
    slot_encoder = sel * (24 * M * d * d + 4 * M * M * d)

    # Decoder per layer: self-attn 8Sd^2 + 4S^2 d; cross-attn q/out 4Sd^2,
    # kv from M slots 4Md^2, scores + AV 4SMd; FFN 16Sd^2.
    decoder = dec * (
        28 * S * d * d
        + 4 * M * d * d
        + 4 * S * S * d
        + 4 * S * M * d
    )

    return Counts({
        # Embedding-side compute: the per-slot spectrum encoder. Part of
        # Porian correction #1 (embeddings are not free).
        "material_encoder": 2 * M * (e_in * eh + eh * eo),
        "slot_proj": 2 * M * eo * d,
        "slot_encoder": slot_encoder,
        "past_proj": 2 * L * eo * d,
        "lab_proj": 2 * 4 * d,
        "decoder": decoder,
        # Output head: fires S * M times per example. Usually the single
        # largest term. This is the crux of correction #1 for INDIGO.
        "thickness_head": 2 * S * M * (2 * d * d + d * NT),
        "eos_head": 2 * S * d,
    })


def _forward_flop_counts_mlp(config) -> Counts:
    d = config.d_model
    e_in = feature_dim(config.feature_mode)
    eh, eo = config.encoder_hidden, config.encoder_out
    input_dim = 3 + M_MAX * eo + M_MAX * MAX_LAYERS + 1
    return Counts({
        "material_encoder": 2 * M_MAX * (e_in * eh + eh * eo),
        "backbone_in": 2 * input_dim * d,
        "backbone_hidden": 2 * (config.n_layers - 1) * d * d,
        "output_head": 2 * d * config.vocab_size,
    })


def forward_flops_per_example(config, **kw) -> float:
    return float(forward_flop_counts(config, **kw).total)


# ============================================================================
# Training FLOPs and budget algebra
# ============================================================================


def train_flops_per_example(config, backward_multiplier: float = 2.0,
                            **kw) -> float:
    """Forward + backward FLOPs for one example."""
    return (1.0 + backward_multiplier) * forward_flops_per_example(config, **kw)


def train_flops(config, n_examples: float, **kw) -> float:
    """Total training FLOPs for `n_examples` example-passes."""
    return train_flops_per_example(config, **kw) * n_examples


def examples_for_budget(config, budget_flops: float, **kw) -> float:
    """Example-passes affordable at `budget_flops`. The IsoFLOP D(N, C)."""
    return budget_flops / train_flops_per_example(config, **kw)


def steps_for_budget(config, budget_flops: float, batch_size: Optional[int] = None,
                     **kw) -> float:
    """Optimizer steps affordable at `budget_flops`."""
    bs = batch_size if batch_size is not None else config.batch_size
    return examples_for_budget(config, budget_flops, **kw) / bs


def budget_for_examples(config, n_examples: float, **kw) -> float:
    """Inverse of `examples_for_budget`; alias of `train_flops`."""
    return train_flops(config, n_examples, **kw)


def tokens_per_example(mean_layers: float = DEFAULT_MEAN_LAYERS) -> float:
    """Supervised prediction targets per example: one per layer, plus EOS."""
    return mean_layers + 1.0


def flops_per_param_per_example(config, **kw) -> float:
    """Diagnostic: the true analogue of the `6` in `C = 6ND`.

    For a language model this is ~6 (per token). Reported per EXAMPLE here,
    so values of 100-130 are expected and are the quantitative statement of
    why `6ND` must not be used.
    """
    return train_flops_per_example(config, **kw) / n_params(config)


# ============================================================================
# Warmup as a FLOP fraction (Porian correction #2)
# ============================================================================


def warmup_steps_for_flop_fraction(config, budget_flops: float,
                                   flop_fraction: float = 0.01,
                                   batch_size: Optional[int] = None,
                                   min_steps: int = 10, **kw) -> int:
    """Warmup length expressed as a fraction of TOTAL TRAINING FLOPs.

    Note for the record: within a single run, FLOPs per step is constant, so
    a fraction of total FLOPs is mathematically identical to a fraction of
    total steps. `scripts/training.py:184` already computes
    `warmup_steps = int(total_steps * warmup_fraction)`, so INDIGO ALREADY
    satisfies Porian correction #2 -- the correction targets Kaplan's use of
    a *fixed* step count (3000), which consumes a pathological fraction of a
    short run.

    The one thing fraction-of-steps does not give us is a floor. At the
    smallest budgets in this sweep a 1-2% warmup is only a handful of steps,
    which makes the schedule degenerate. `min_steps` supplies that floor,
    and is the only change the sweep needs on this axis.
    """
    total = steps_for_budget(config, budget_flops, batch_size, **kw)
    return max(min_steps, int(round(total * flop_fraction)))


# ============================================================================
# Verification
# ============================================================================


def test_no_grad_undercount_regression() -> bool:
    """Assert the documented `FlopCounterMode` blind spot still behaves as described.

    This is a regression guard on our *understanding*, not on our code. If a
    torch upgrade fixes the fused-path instrumentation, this test fails loudly
    and the module docstring needs updating -- rather than us quietly keeping
    a stale warning, or worse, someone re-deriving the compute axis from a
    `no_grad` measurement and getting numbers 2.7x too small.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from src.model import ModelConfig, build_model

    cfg = ModelConfig(head_mode="cross_attn", d_model=1024, n_heads=8,
                      slot_encoder_layers=4, decoder_layers=1, dropout=0.0)
    B = 4
    args = (
        torch.randn(B, 3),
        torch.randn(B, M_MAX, 2, feature_dim(cfg.feature_mode) // 2),
        torch.ones(B, M_MAX, dtype=torch.bool),
        torch.zeros(B, M_MAX, MAX_LAYERS),
        torch.full((B,), M_MAX),
    )

    def slot_encoder_flops(training: bool, no_grad: bool) -> float:
        # A fresh model per cell: the fast-path decision must not be
        # inherited from a previous forward.
        model = build_model(cfg)
        model.train(training)
        counter = FlopCounterMode(display=False, depth=2)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with counter, ctx:
            model(*args)
        return sum(
            counter.get_flop_counts()
            .get("FlexMaterialCrossAttn.slot_encoder", {})
            .values()
        )

    matrix = {
        (training, no_grad): slot_encoder_flops(training, no_grad)
        for training in (True, False)
        for no_grad in (False, True)
    }
    # Exactly one cell -- eval + no_grad -- should lose the slot encoder.
    expected_zero = {(False, True)}
    got_zero = {k for k, v in matrix.items() if v == 0}
    ok = got_zero == expected_zero
    if not ok:
        print("  [regression] FlopCounterMode fused-path behaviour CHANGED.")
        for (training, no_grad), v in sorted(matrix.items()):
            print(f"      training={training!s:5s} no_grad={no_grad!s:5s} "
                  f"slot_encoder={v:.4e}")
        print("      Update the flops.py docstring to match.")
    else:
        print("  fused-path blind spot present as documented "
              "(slot_encoder = 0 only under eval + no_grad)")
    return ok


def verify_against_torch(configs=None, verbose: bool = True) -> bool:
    """Check params exactly and FLOPs to ~2% against torch.

    Requires torch. Runs with gradients enabled deliberately: see the module
    docstring on `FlopCounterMode` reporting zero FLOPs for the fused encoder
    path under `torch.no_grad()`.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from src.model import ModelConfig, build_model

    if configs is None:
        configs = [
            # Production baseline. Must come out at 69.5749M.
            ModelConfig(head_mode="cross_attn", d_model=1024, n_heads=8,
                        slot_encoder_layers=4, decoder_layers=1),
            ModelConfig(head_mode="cross_attn", d_model=512, n_heads=8,
                        slot_encoder_layers=4, decoder_layers=1),
            ModelConfig(head_mode="cross_attn", d_model=256, n_heads=4,
                        slot_encoder_layers=2, decoder_layers=1),
            ModelConfig(head_mode="cross_attn", d_model=128, n_heads=2,
                        slot_encoder_layers=2, decoder_layers=1),
            ModelConfig(head_mode="cross_attn", d_model=64, n_heads=1,
                        slot_encoder_layers=1, decoder_layers=1),
            ModelConfig(head_mode="cross_attn", d_model=1024, n_heads=8,
                        slot_encoder_layers=8, decoder_layers=2),
            ModelConfig(head_mode="mlp", d_model=1024, n_layers=8),
        ]

    B = 4
    ok = True
    if verbose:
        print(f"{'config':<34} {'N (torch)':>12} {'N (calc)':>12} "
              f"{'fwd (torch)':>12} {'fwd (calc)':>12} {'err':>7}")
    for cfg in configs:
        model = build_model(cfg)
        n_torch = sum(p.numel() for p in model.parameters())
        n_calc = n_params(cfg)
        if n_torch != n_calc:
            ok = False
            print(f"  PARAM MISMATCH for {cfg.head_mode} d={cfg.d_model}: "
                  f"torch={n_torch:,} calc={n_calc:,}")

        # Dropout off and train mode: dropout would not change FLOPs, but
        # train mode is required to defeat the fused encoder fast path.
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

        lab = torch.randn(B, 3)
        pf = torch.randn(B, M_MAX, 2, feature_dim(cfg.feature_mode) // 2)
        pm = torch.ones(B, M_MAX, dtype=torch.bool)
        sm = torch.zeros(B, M_MAX, MAX_LAYERS)
        sm[:, 0, :3] = 1.0
        ps = torch.full((B,), M_MAX)

        counter = FlopCounterMode(display=False)
        with counter:
            model(lab, pf, pm, sm, ps)
        fwd_torch = counter.get_total_flops() / B
        # The counter measures a single packed forward for cross_attn and a
        # single fanned-out step for mlp, so compare against packed=True.
        fwd_calc = forward_flops_per_example(cfg, packed=True)
        err = (fwd_calc - fwd_torch) / fwd_torch if fwd_torch else 0.0
        if abs(err) > 0.03:
            ok = False
            print(f"  FLOP MISMATCH > 3% for {cfg.head_mode} d={cfg.d_model}: "
                  f"{err:+.2%}")
        if verbose:
            name = (f"{cfg.head_mode} d{cfg.d_model} "
                    f"se{slot_encoder_depth(cfg)} dec{cfg.decoder_layers}")
            print(f"{name:<34} {n_torch:>12,} {n_calc:>12,} "
                  f"{fwd_torch:>12.4e} {fwd_calc:>12.4e} {err:>+6.2%}")
    return ok


# ============================================================================
# CLI
# ============================================================================


def _main() -> None:
    import argparse

    from src.model import ModelConfig

    p = argparse.ArgumentParser(
        description="INDIGO parameter / FLOP calculator (scaling study Phase 0.1)")
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--slot-encoder-layers", type=int, default=4)
    p.add_argument("--decoder-layers", type=int, default=1)
    p.add_argument("--encoder-hidden", type=int, default=128)
    p.add_argument("--encoder-out", type=int, default=64)
    p.add_argument("--feature-mode", type=str, default="raw_spectrum")
    p.add_argument("--head-mode", type=str, default="cross_attn",
                   choices=["cross_attn", "mlp"])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--budget", type=float, default=None,
                   help="Compute budget in FLOPs; prints the implied D and steps.")
    p.add_argument("--corpus", type=float, default=None,
                   help="Corpus size in examples; prints implied epochs.")
    p.add_argument("--verify", action="store_true",
                   help="Validate params/FLOPs against torch and exit.")
    args = p.parse_args()

    if args.verify:
        ok = verify_against_torch()
        print()
        ok = test_no_grad_undercount_regression() and ok
        print("\nOK" if ok else "\nFAILED")
        raise SystemExit(0 if ok else 1)

    cfg = ModelConfig(
        head_mode=args.head_mode, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, slot_encoder_layers=args.slot_encoder_layers,
        decoder_layers=args.decoder_layers, encoder_hidden=args.encoder_hidden,
        encoder_out=args.encoder_out, feature_mode=args.feature_mode,
        batch_size=args.batch_size,
    )

    pc = param_counts(cfg)
    fc = forward_flop_counts(cfg)
    print(f"config: {cfg.head_mode} d_model={cfg.d_model} "
          f"slot_encoder_layers={slot_encoder_depth(cfg)} "
          f"decoder_layers={cfg.decoder_layers} n_heads={cfg.n_heads}")
    print(f"\nPARAMETERS  (total {pc.total:,} = {pc.total / 1e6:.4f}M)")
    print(pc.table(1e6, "M"))
    print(f"\nFORWARD FLOPs PER EXAMPLE  (total {fc.total:.4e})")
    print(fc.table(1e9, "G"))
    print(f"\ntrain FLOPs/example        {train_flops_per_example(cfg):.4e}")
    print(f"FLOPs/param/example        {flops_per_param_per_example(cfg):.1f}"
          f"   (the '6' in 6ND is {flops_per_param_per_example(cfg) / 6:.0f}x too small)")
    print(f"tokens/example             {tokens_per_example():.1f}")

    if args.budget:
        D = examples_for_budget(cfg, args.budget)
        steps = steps_for_budget(cfg, args.budget)
        print(f"\nAT BUDGET C = {args.budget:.3e} FLOPs")
        print(f"  D                        {D:,.0f} examples "
              f"({D * tokens_per_example():,.0f} tokens)")
        print(f"  steps @ bs={cfg.batch_size:<5}         {steps:,.0f}")
        print(f"  warmup steps @ 1% FLOPs  "
              f"{warmup_steps_for_flop_fraction(cfg, args.budget)}")
        if args.corpus:
            print(f"  epochs over {args.corpus:,.0f}    {D / args.corpus:.2f}")


if __name__ == "__main__":
    _main()
