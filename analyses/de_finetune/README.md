# ΔE Post-Training Finetune

**Status:** design (Sept 2026). This document is the design specification;
implementation lives in `src/de_finetune.py`, `src/optical_sim_diff.py`,
and `slurms/finetune_de.sh`.

**Analogy.** Modern LLMs use a two-stage recipe: **pretrain** on
next-token cross-entropy over a large corpus, then **finetune / align**
against the actual downstream objective (instruction-following, helpfulness,
preference). INDIGO's current training pipeline covers only the first
stage. This document describes the second stage: a short, targeted
**post-training finetune** that optimises the model directly against
ΔE₀₀ — the perceptual color-difference metric INDIGO is ultimately deployed
to minimise.

If it works, the finetuned model should produce **greedy-decoded
structures whose achieved Lab is close to the target**, without depending
on the inference-time sample-and-refine loop that currently bridges the
gap between the pretrained model's token argmax and the actual optical
objective.

---

## 1. Motivation — the CE / ΔE mismatch

The pretraining loss is per-token cross-entropy over a 3201-way
vocabulary (32 material slots × 100 thickness bins + EOS). Two problems:

- **Token error and ΔE are only loosely correlated.** Two thicknesses
  1 nm apart can produce either ΔE ≈ 0.3 (indistinguishable) or ΔE ≈ 8
  (highly visible) depending on the interference regime. CE treats them
  identically (both wrong tokens, 1 unit of loss each).
- **The model's argmax is over-confident.** With no label smoothing, the
  softmax over 3201 tokens learns very peaked distributions. The tier
  A/B eval below shows T=0 (greedy) actually *beats* T=1 (sampled) on ΔE
  — sampling adds noise the peaked distribution can't compensate for.

**Tier A/B baseline (3-epoch pretrain, `latest/`):**

| | Tier A | | Tier B | |
|---|---|---|---|---|
| | T=0 greedy | T=1 sample | T=0 greedy | T=1 sample |
| Overall ΔE mean | 15.71 | 18.14 | 12.70 | 14.58 |
| Overall ΔE median | 11.80 | 13.94 | 7.94 | 9.92 |
| HC ΔE mean | 20.65 | 23.28 | 18.20 | 20.18 |
| Random ΔE mean | 13.86 | 16.22 | 10.80 | 12.64 |

Everything is well above the ΔE = 2 perceptibility threshold. HC
structures are ~2× harder than random ones. The training loss can't
close this — it's optimising the wrong thing.

**The fix:** a post-training stage that backprops through ΔE itself.

---

## 2. High-level recipe

Load the pretrained checkpoint (val-optimal step, not `latest/`), then
run a short finetune with the following per-step loop:

```
for each training example (target_lab, pool, ground_truth_structure):
    k = uniform{1..N}                                     # pick a layer position
    prefix = ground_truth[0..k-1]                          # teacher-forced GT
    suffix = ground_truth[k+1..N-1]                        # teacher-forced GT (detached)
    logits_k = model(target_lab, pool, prefix)             # differentiable
    slot_k, thickness_k = STE(logits_k)                    # forward=hard, backward=soft
    full_stack = prefix + [(slot_k, thickness_k)] + suffix # length N
    predicted_lab = differentiable_sim(pool, full_stack)   # via torch↔JAX wrapper
    loss = ciede2000(target_lab, predicted_lab)
    loss.backward()
```

Gradient flows only through layer k's logits. Layers before and after
are frozen ground-truth thicknesses/slots that just contribute to the
full-stack physics.

### Why "layer k differentiable, others GT" (credit assignment)

If we sim'd only the prefix `[0..k-1] + predicted_k`, the model would
be penalised for missing the full target Lab with a partial stack of
`k` layers — an unfair signal because the target was defined by an
N-layer stack. In interference physics, a 5-layer subset of an 8-layer
design almost never produces the same Lab as the full stack.

By keeping layers `k+1..N-1` at their ground-truth values, the sim
evaluates the physically-correct question: **given the surrounding
stack, what's the best choice at position k?** This is the same
credit-assignment fix used in sequence-level RL for translation (BLEU
computed on full sequence, credit assigned per token), chain-of-thought
reasoning (grade only the final answer, credit intermediate steps by
their contribution), and TTS decoders (perceptual loss on full waveform,
not per-frame).

Every 8-layer structure yields 8 training examples (one per k position).
Every training example evaluates the full-stack ΔE₀₀ — the deployed
objective, at the granularity it's defined.

### Training-inference gap

At inference the model doesn't have GT layers `k+1..N-1`; it predicts
them autoregressively. Our training uses their GT values. This is the
same as standard teacher-forced seq2seq training and works because the
pretrained model's own future rollouts are usually close to GT for a
well-trained backbone. If the finetune plateaus while inference-time
ΔE remains meaningfully worse than training-time ΔE, upgrade to
autoregressive-suffix rollout as v2 — more expensive, higher variance,
but bridges the gap.

---

## 3. Making it differentiable

Three technical pieces stitch this together.

### 3.1 Straight-through estimator (STE) on both slot and thickness

The pretrained model emits categorical logits over slots × thickness
bins. Two questions:

| Output | Physical semantics | Differentiability trick |
|---|---|---|
| **Material slot** | Categorical. Cannot physically interpolate — averaging two n/k spectra is nonsense. | **STE on argmax slot.** Forward: hard one-hot. Backward: softmax gradient. |
| **Thickness bin** | Interference-optical. **The posterior can be multimodal**: 10 nm and 100 nm can both produce the target color, but 55 nm can produce a very different color. Averaging bins is unsafe. | **STE on argmax bin.** Forward: hard bin center in nm. Backward: softmax gradient over bins, ordinally aware via `bin_centers`. |

**Aside — why thickness is a classifier at all (pretrain design choice).**
A regression head over a continuous thickness scalar would be the natural
first instinct. It's not what we do, because the posterior over
`thickness | (target_lab, prefix)` is genuinely multimodal in
interference optics. For a given target color and prefix, thicknesses
of ~10 nm and ~100 nm can both hit ΔE ≈ 1, while the arithmetic mean
(~55 nm) hits ΔE ≈ 8. A regressor trained on both examples would learn
to output 55 — the mean of the two correct answers is a wrong answer.
A classifier over bins can put probability on both modes and pick either
one at decode time; the multimodality is preserved. Same reason we
STE-on-argmax-bin here rather than passing a soft-averaged thickness to
the sim — soft averaging would recreate the "mean of two good answers"
problem inside every training step.

Both use the identical trick: hard forward, soft backward, no averaging
of physically-incompatible outputs. The forward pass always presents the
simulator with a single-material, single-thickness stack — a physically
valid design that could actually be manufactured.

**One-layer worked example** (batch dim omitted for readability):

```python
import torch, torch.nn.functional as F

NUM_THICKNESSES = 100
M_MAX = 32
THICKNESS_BIN_CENTERS = torch.linspace(5.0, 200.0, NUM_THICKNESSES)  # [100]

# Model output for one decoding step: 3201-token vocab.
raw_logits = model_output                                             # [3201]
thickness_logits = raw_logits[:M_MAX * NUM_THICKNESSES].view(
    M_MAX, NUM_THICKNESSES,
)

# --- Slot: STE over argmax slot ---
slot_logits = thickness_logits.max(dim=-1).values                     # [M_MAX]
slot_probs = F.softmax(slot_logits, dim=-1)                           # [M_MAX]
slot_hard_idx = slot_probs.argmax()
slot_onehot_hard = F.one_hot(slot_hard_idx, M_MAX).float()            # [M_MAX]
slot_choice = slot_onehot_hard + slot_probs - slot_probs.detach()     # STE

# --- Thickness: STE over argmax bin, conditioned on chosen slot ---
chosen_thick_logits = (slot_choice.unsqueeze(-1) * thickness_logits).sum(0)
thick_probs = F.softmax(chosen_thick_logits, dim=-1)
thick_hard_idx = thick_probs.argmax()
thick_onehot_hard = F.one_hot(thick_hard_idx, NUM_THICKNESSES).float()
thick_onehot_ste = thick_onehot_hard + thick_probs - thick_probs.detach()
thickness_nm = (thick_onehot_ste * THICKNESS_BIN_CENTERS).sum()       # scalar

# Both `slot_choice` (one-hot, STE'd) and `thickness_nm` (scalar, STE'd)
# are differentiable w.r.t. `raw_logits`, but the forward pass presents
# hard, physically valid values to the simulator.
```

**What STE looks like in one line:** `x_hard + x_soft - x_soft.detach()`.
Forward: the `x_soft` and `-x_soft.detach()` cancel (both are the same
value; only one carries gradient), leaving `x_hard`. Backward: the
gradient of `x_soft` flows (the other two terms are constants w.r.t.
logits). Standard trick from Bengio et al. 2013; also underlies
VQ-VAE, Gumbel-Softmax variants, and every discrete-latent finetune
setup.

### 3.2 Differentiable optical simulator (PyTorch ↔ JAX bridge)

The optical simulator (`src/optical_sim.py`) uses `jaxlayerlumos` for
the transfer-matrix calculation. JAX is differentiable end-to-end and
gives us `jax.vjp` out of the box, so no reimplementation is needed.
The only wiring is a PyTorch autograd bridge:

```python
class DifferentiableStackSim(torch.autograd.Function):
    """
    Forward:  torch tensors -> convert to JAX -> stackrt_n_k -> convert to torch.
    Backward: use jax.vjp on the same computation to get gradients w.r.t.
              inputs (thicknesses; optionally n_k features).
    """
```

Inputs:
- `n_stack, k_stack` — per-wavelength complex material features assembled
  from the slot choice (one_hot-weighted sum over pool_features so
  gradients flow through the slot STE).
- `thicknesses_nm` — differentiable scalars from the thickness STE.

Output:
- `predicted_lab` — [3] tensor, differentiable w.r.t. both inputs.

Numerically identical to the existing forward pass; the only new
capability is the backward.

### 3.3 Full-stack rollout module

`src/de_finetune.py` wraps the pieces above with:
- Per-example layer sampling (`k ~ Uniform{1..N}`).
- Teacher-forced prefix + STE-predicted layer k + teacher-forced suffix.
- One sim call per example, one ΔE loss.
- Monitoring hooks (slot-argmax-flip rate, thickness entropy).

### 3.4 How the gradient reaches every (material, thickness) combo

A common intuition trap: since the sim is called once per (example, k)
with a single hard slot and a single hard thickness bin, it looks like
only *that* one combo can be improved. In fact **every entry of the
3201-way vocab receives a gradient at position k**, via a first-order
linear projection of the sim's analytic gradient. Understanding this
projection is the whole reason we're not paying a 3201× sim overhead.

**One sim, three tensors of gradient.** `jax.vjp` gives us the analytic
gradient at the single simulated point:

```
∂ΔE/∂n_layer_k[λ],   ∂ΔE/∂k_layer_k[λ],   ∂ΔE/∂thickness_k
```

(vectors of length `NUM_LAMBDA` and one scalar, all evaluated at the
argmax combo we actually simulated).

**Projection to every slot.** The layer-k n/k tensor is built as
`n_layer_k = slot_choice @ pool_n` where `slot_choice` is the STE'd
one-hot. In the backward, that linear map turns the sim gradient into
a per-slot gradient:

```
∂ΔE/∂slot_choice[i]  ≈  ⟨∂ΔE/∂n_layer_k, pool_n[i]⟩
                       + ⟨∂ΔE/∂k_layer_k, pool_k[i]⟩
```

That's a scalar per slot `i`: *"how does ΔE change if I nudge the
current n/k a tiny bit toward material i's spectrum?"* It's not a
literal simulation of material i — it's the sim's *local* gradient
projected onto material i's spectral profile.

**Projection to every thickness bin.** Same story with
`thickness_nm = thick_choice @ bin_centers`:

```
∂ΔE/∂thick_choice[j]  ≈  ∂ΔE/∂thickness_k · bin_centers[j]
```

*"How does ΔE change if I nudge thickness toward bin j's center,
using the local ∂ΔE/∂t I already have."*

Then both per-vocab gradients flow through the softmax, back through
the transformer, into every trainable parameter.

**What this buys us (and where it can bite).**

- **Cheap:** 1 sim per (example, k), not 3201. This is what makes 1M
  examples × ~7 layers avg = ~7M sims per epoch tractable.
- **Locally correct:** for materials whose n/k is close to the current
  pick, or bins near the current thickness, the linearization is
  accurate.
- **Globally biased (classic STE):** for a material with wildly
  different n/k, the linear projection can point in the wrong direction
  ("looks like it would help" — but a full sim there might disagree).
  This is the standard STE bias.
- **Self-correcting under small LR:** as the argmax pick shifts during
  training, the anchor point of the linearization shifts with it. Small
  steps keep us in the locally-honest regime, and each step re-anchors.
  This is why LR is 5e-6 / 1e-6 here — small enough that no single update
  moves us out of the linear neighbourhood the sim gradient was valid in.

**Mental model summary.** For each layer position, the sim runs once
at the argmax combo, and its analytic gradient is *projected* onto every
material and every thickness bin via their spectral profile and bin
position. Every vocab entry gets a first-order linear estimate of
"would picking me lower ΔE?" — cheap and locally-honest, with the STE
bias absorbed by small LR + short finetune.

---

## 4. Two experiments

Both start from the **val-optimal pretrained checkpoint** (step ~13k
under the current 3-epoch run — *not* `latest/`, which is past the
overfit inflection).

### Experiment A — decoder-only finetune

- **Frozen:** `MaterialEncoder` (per-slot conv/MLP over material
  spectra) + `slot_encoder` (transformer over pool of size M).
- **Trainable:** `decoder` (causal self-attn + cross-attn) +
  `thickness_head` + `eos_head`.
- **Rationale:** the pretrained encoder already knows how to map n/k
  spectra to slot embeddings. Post-training only needs to realign the
  "given the slot embeddings, pick the right thickness" part.
- **LR:** `5e-6`, cosine to zero, 2% warmup.
- **Loss:** pure ΔE₀₀ (no CE co-loss). Frozen encoder anchors the
  representations.

### Experiment B — full-model finetune

- **Frozen:** nothing.
- **Trainable:** everything.
- **Rationale:** maybe the pretrained slot representations don't
  distinguish materials that produce visually similar Lab well enough
  — letting the encoder shift could raise the ceiling.
- **LR:** `1e-6`, cosine to zero, 2% warmup. (Smaller than A because
  more params to move, more capacity to forget the pretrain.)
- **Loss:** pure ΔE₀₀. If catastrophic forgetting is observed
  (pretrain CE loss rises sharply during finetune), fall back to
  `L = ΔE + 0.1·CE`. Monitored via a periodic CE eval on a held-out
  slice.

**Order of operations:** run A first. If A doesn't improve greedy-ΔE
meaningfully, don't bother with B (the additional trainable capacity
is unlikely to help if decoder-only can't). If A improves, run B for
the ceiling.

---

## 5. Data

Fresh **1M examples with `high_chroma_prob = 0.30`**, generated by
`slurms/generate_finetune_data.sh` to a distinct directory
(`data/finetune/`, angle-substrate layout identical to `data/train/`).

Why fresh vs oversampling the existing 10M:

- **Fresh Lorentz materials.** New synthetic material draws — the
  post-training set doesn't share the pretrain's random-material
  distribution. A cleaner test of "does the model generalise post-
  training or just re-fit the pretrain distribution."
- **Distribution match to the objective.** The pretrain was 15% HC; if
  we finetune on the pretrain distribution, we implicitly under-weight
  HC (the hardest and most valuable). Fresh generation with 30% HC
  changes both the ratio and the specific hard cases the model sees.
- **Independent val slice.** With `--split all` on the tier evals we
  already have canonical held-out test sets; the finetune data doesn't
  need to reserve a large val split. We use 5k (0.5%) for finetune val
  loss tracking.

Why 1M vs 10M:

- The finetune isn't learning representations — it's realigning them.
  LLM post-training stages are typically 10-100× smaller than pretrain
  for the same reason.
- Each ΔE step is ~10-100× more expensive than a CE step (sim in
  forward pass). Compute-matched, 1M finetune ≈ 10M pretrain steps.
- Smaller finetune sets afford ablations: two experiments (A/B) × a few
  LR settings = ~6 runs, each ~1 day on GPU-JAX. 10M would be a month.

---

## 6. Hyperparameters and schedule

| Knob | Experiment A | Experiment B |
|---|---|---|
| LR | 5e-6 | 1e-6 |
| Warmup fraction | 2% | 2% |
| LR schedule | cosine to 0 | cosine to 0 |
| Epochs | 3 | 3 |
| Batch size | 128 | 128 |
| Optimizer | AdamW (wd=0.01) | AdamW (wd=0.01) |
| Grad clip | 1.0 | 1.0 |
| Loss | pure ΔE₀₀ | pure ΔE₀₀ (fallback: +0.1·CE) |
| Rollout | full-stack, layer k differentiable | same |
| Data | data/finetune (1M, HC=0.30) | same |

Batch size is smaller than pretrain (128 vs 512) because the sim call
is the throughput bottleneck; smaller batches keep the pipeline
utilised.

---

## 7. Monitoring

Every step logs to `history.jsonl` (same format as pretrain):

- `loss_de` — mean ΔE₀₀ across the batch (both training signal and
  primary metric).
- `slot_argmax_flip_rate` — fraction of examples where the argmax
  slot differed between adjacent steps for the same k-position. High
  values (>5%) signal top-2 slot instability; if sustained, add a small
  entropy penalty on slot logits.
- `thickness_entropy_nats` — mean entropy of the thickness softmax.
  Values below ~0.5 nats mean the thickness posterior has collapsed
  to a single bin (gradients vanish); add a small entropy floor if
  observed.
- `lr` — current learning rate (schedule progress).

Periodic (every N steps) full-eval on a 5k val slice with the same
metrics we log at pretrain: overall / HC / random ΔE₀₀
mean/median/p75.

Auto-fire at end of run: `plot_training_curve.sh` on the finetune
history, then `de_curve.sh` sweeping the checkpoints from step 0
(= pretrain init) through the finetune, showing whether ΔE₀₀ actually
falls monotonically.

---

## 8. What we're watching for

**Success looks like:**

- Greedy-decoded T=0 ΔE₀₀ mean on tier A drops from ~15.7 toward ~5-10.
- HC / random gap narrows (currently 2×; the HC-heavy finetune should
  help even before any architectural change).
- T=0 catches or beats T=1-plus-refinement (the whole point).

**Warning signs:**

- Slot-flip rate climbs above 5% and doesn't damp → soft distributions
  are too flat; either raise LR, add entropy penalty, or the pretrain
  representations are noisier than we thought.
- Thickness entropy floors near zero → posterior collapse. Add entropy
  regulariser or reduce LR.
- Full-model experiment: pretrain CE loss rises sharply on a held-out
  slice → catastrophic forgetting; switch to `L = ΔE + 0.1·CE`.
- Finetune val ΔE plateaus at a value close to greedy tier-A baseline
  → the CE-vs-ΔE gap wasn't the bottleneck; likely a capacity/data
  limit, and a bigger pretrain (or different architecture) is warranted.

---

## 9. Provenance and future work

- **Provenance:** this stage was added after tier A/B evals at T=0 and
  T=1 showed the ΔE greedy floor at ~12-16 mean and revealed the CE /
  ΔE gap. Design conversation is in the session log; the fix is
  documented above.
- **v2 (if needed):** autoregressive-suffix rollout — predict layers
  `k+1..N-1` under STE from layer k's prediction (not from GT), sim,
  backprop. Bridges the training/inference gap but ~N× the per-step
  compute and higher variance. Only pursue if v1 plateaus while the
  train/test ΔE gap remains meaningful.
- **v3 (further out):** joint slot + thickness refinement via
  autoregressive generation with all decoder layers STE'd — closest to
  end-to-end but expensive. Consider only after v1/v2 land clean
  results.
