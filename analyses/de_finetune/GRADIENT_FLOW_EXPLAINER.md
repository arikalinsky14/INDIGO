# ΔE Finetune — Gradient Flow, Explained From Scratch

A self-contained walkthrough of how the post-training stage gets a gradient
from a physics simulator back into a transformer's weights. About a 30-min
read, or 20 min of Q&A with a teaching agent (instructions at the bottom).

The formal design lives in [`README.md`](./README.md). This document is
the pedagogical counterpart: same content, chronological, one trick at
a time, with the *why* spelled out for each.

---

## Prerequisites

You should be comfortable with:

- **Transformers** at the "logits over a vocab per position" level.
- **Backprop / autograd**: what `.backward()` does, what `.detach()` does,
  why `requires_grad` matters.
- **INDIGO pretraining at a high level**: the model takes a target Lab
  color and a pool of candidate materials and predicts a sequence of
  layer picks (material, thickness). Trained on token-level cross-entropy
  against ground-truth thin-film stacks that were generated in reverse
  (structure → simulate → Lab).

You don't need to know JAX, autograd internals, or the CIEDE2000 formula.

---

## The one-sentence problem

> The metric we care about is ΔE₀₀ between the simulated Lab of the
> predicted structure and the target Lab; the metric we trained against
> is per-token cross-entropy over 3201 tokens. Those two are only
> loosely correlated, so the pretrained model's greedy decode leaves
> ΔE well above human-perceptibility (mean ~15). Post-training fixes
> this by backpropping ΔE₀₀ itself.

Three obstacles stand between us and doing that:

1. **The optical sim lives in JAX, the model lives in torch.** They have
   different autograd systems. We need a bridge.
2. **The model's outputs are discrete.** It picks one of 32 materials
   and one of 100 thickness bins per layer. You can't backprop through
   `argmax`. You also can't just soft-average the picks — see the
   "why bins at all" discussion below.
3. **The loss is a full-stack metric, but we want per-position credit.**
   ΔE is defined on the finished N-layer design. But we're predicting
   one layer at a time. If we sim just a partial stack we get an unfair
   signal; if we STE-pick every layer we accumulate noise.

Every trick below exists to work around one of these.

---

## The pieces

### 1. The pretrained model — what comes in, what comes out

```
inputs:
    target_lab          [B, 3]                  — the color we want
    pool_features       [B, M_max=32, 2, NUM_LAMBDA]
                        (n and k spectra for each material in the pool)
    structure_matrix    [B, MAX_LAYERS+1, ...] — teacher-forced GT layers,
                        used only for the causal context at training time

output:
    logits              [B, MAX_LAYERS+1, VOCAB=3201]
```

`VOCAB = 3201 = M_max (32) × NUM_THICKNESSES (100) + 1 EOS`.

So at each of `MAX_LAYERS+1` positions, the model emits a probability
distribution over "which (material, thickness) pair to place, or stop."

At training time, `structure_matrix` carries the *ground-truth* structure
and the causal mask keeps position `k` from peeking at layers `≥ k`.
So one forward pass gives us N logit vectors per example, one per
decoding step, all consistent with each other.

**Key mental model:** the transformer emits logits at every position in
one shot; we then interpret those logits position by position.

---

### 2. The optical simulator — the physics forward

`OpticalSimulator` (in `src/optical_sim.py`) wraps `jaxlayerlumos`'s
transfer-matrix method. You feed it:

```
n_stack           [N_layers, NUM_LAMBDA]   — real refractive index per λ
k_stack           [N_layers, NUM_LAMBDA]   — extinction coefficient per λ
thicknesses_nm    [N_layers]               — layer thicknesses
```

and it returns the CIE Lab color of the reflected light. This is the
same simulator that generates the pretrain dataset — a compact TMM →
reflectance → CIE XYZ → Lab pipeline.

For pretraining, we only ever called this in the *forward* direction —
never needed a gradient. Now we do.

---

### 3. **TRICK #1 — the torch ↔ JAX bridge (`src/optical_sim_diff.py`)**

**The gap:** the sim is written in JAX; the model is in torch. Torch's
autograd doesn't know how to trace through JAX ops, and vice versa.

**The insight:** JAX has `jax.vjp` — reverse-mode differentiation
built in. Given a function `f(x)` and inputs `x`, `jax.vjp` returns
`(f(x), vjp_fn)` where `vjp_fn(dy)` computes `dy · ∂f/∂x`. That's
*exactly* the interface torch's `torch.autograd.Function` expects for
`backward`.

**The trick:**

```python
class DifferentiableStackSim(torch.autograd.Function):

    @staticmethod
    def forward(ctx, n_stack, k_stack, thicknesses_nm, incidence_angle):
        # 1. torch tensor -> jax array (view, no copy where possible)
        n_jax, k_jax, t_jax = _to_jax(n_stack), _to_jax(k_stack), _to_jax(thicknesses_nm)

        # 2. jax.vjp: get output AND a callable for the reverse pass
        lab_jax, vjp_fn = jax.vjp(_forward_lab, n_jax, k_jax, t_jax, incidence_angle)

        # 3. stash vjp_fn on ctx for backward
        ctx.vjp_fn = vjp_fn

        # 4. return output as torch tensor
        return _to_torch(lab_jax)

    @staticmethod
    def backward(ctx, grad_output):
        # 1. torch grad -> jax
        grad_jax = _to_jax(grad_output)

        # 2. apply the saved vjp: pushes gradient back to inputs
        gn_jax, gk_jax, gt_jax, _ = ctx.vjp_fn(grad_jax)

        # 3. jax -> torch and return in the same order forward's inputs came in
        return _to_torch(gn_jax), _to_torch(gk_jax), _to_torch(gt_jax), None
```

**What this buys us:** for the outside torch world, `differentiable_compute_lab(n, k, t)`
looks like an ordinary torch op that returns Lab and knows how to
compute `∂Lab/∂n`, `∂Lab/∂k`, `∂Lab/∂t`. Behind the scenes JAX does the
heavy lifting.

**How we verified it:** `src/optical_sim_diff.py`'s `__main__` runs a
smoke test that (a) checks the forward matches the pretrain numpy path
bit-for-bit (Lab diff < 1e-3 after some CIE-pipeline float drift), and
(b) checks the analytic thickness gradient from `jax.vjp` against a
finite-difference numeric estimate (max error < 1e-3).

---

### 4. **TRICK #2 — Straight-through estimator (STE) on the discrete picks**

**The gap:** the model outputs categorical logits. To feed the sim we
need one concrete material and one concrete thickness — not a soft
average. But `argmax` has no gradient.

**Why we can't just soft-average:**

- **Material:** you can't linearly interpolate two n/k spectra and get a
  physical material. Titanium ⊕ chromium is not a real substance; its
  n/k has no manufacturable meaning.
- **Thickness:** the posterior is genuinely *multimodal*. For a given
  target color, thicknesses of ~10 nm and ~100 nm can both give
  ΔE ≈ 1, but the arithmetic mean of ~55 nm gives ΔE ≈ 8. If we
  soft-averaged, we'd feed the sim a value neither mode endorses.

**The trick (Bengio et al. 2013):**

```python
def _ste_onehot(soft_probs):
    hard_idx = soft_probs.argmax(dim=-1)
    hard_onehot = F.one_hot(hard_idx, num_classes=...).to(dtype=soft_probs.dtype)
    return hard_onehot + soft_probs - soft_probs.detach()
```

Let's trace forward and backward:

- **Forward:** `soft_probs - soft_probs.detach()` equals `soft_probs - soft_probs`
  numerically (`.detach()` returns the same value, just without gradient).
  That's 0. Result: `hard_onehot + 0 = hard_onehot`. The sim sees a hard pick.
- **Backward:** gradient of `hard_onehot` w.r.t. logits = 0 (it's derived
  through argmax). Gradient of `-soft_probs.detach()` = 0. Gradient of
  `soft_probs` = the real softmax gradient. So total = softmax gradient.
  The transformer sees a soft update.

**Result:** the sim gets a physically valid input, the model gets a
gradient as if we'd used the soft distribution.

Applied twice in `ste_pick`: once for slot, once for thickness bin.
Thickness gets the extra step `thickness_nm = thick_choice @ bin_centers`,
which is just a linear map from a one-hot to a scalar in nm.

---

### 5. **TRICK #3 — Full-stack rollout with per-position credit assignment**

**The gap:** ΔE is only meaningful on a *complete* N-layer stack (physics
of interference depends on all layers). But we want to differentiate one
position at a time — otherwise we'd have to STE every layer and stack
errors would compound.

**The wrong approach:** at position k, sim only `layers[0..k]` (prefix
+ our prediction). Then the loss punishes the model for not hitting the
target with an incomplete stack. That's unfair — the target was defined
by an N-layer stack.

**The right approach (credit assignment):** at position k, sim the FULL
N-layer stack, but only layer k has `requires_grad=True`. The others
are teacher-forced ground-truth values with `.detach()`:

```python
n_stack_full = torch.cat([
    gt_n_stack[:k].detach(),          # prefix — GT, no gradient
    n_layer_k.unsqueeze(0),           # THIS layer — differentiable STE pick
    gt_n_stack[k + 1:].detach(),      # suffix — GT, no gradient
], dim=0)
```

The sim gets physically correct inputs (full stack), but the ΔE loss
only distributes gradient into layer k. The question the loss asks is:

> *Given that every other layer is exactly right, was your pick at
> position k the best possible pick?*

That's the correct per-position credit assignment. Iterate `k = 0..N-1`
and every position gets exactly one such gradient contribution per
training pass through the example.

**Analogies from other ML domains:**

- **Sequence-level RL for translation:** BLEU is defined on the full
  sentence, but credit is assigned per token.
- **Chain-of-thought training:** grade only the final answer, assign
  credit to intermediate reasoning steps.
- **TTS decoders:** perceptual loss on the full waveform, but the
  decoder is trained per frame.

---

### 6. **TRICK #4 — `-inf` sanitization (the bug we just fixed)**

Not a trick from the literature — a numerical trap we hit and had to
guard against. Worth documenting because it's the class of bug that
kills silently.

**The gap:** the model is called with `apply_output_mask=True`, which
writes `-inf` into logits for padded pool slots (pools have variable
size ≤ 32; the extras are masked out). The slot STE handles `-inf`
fine: `softmax([-inf, real, real, ...])` drops the `-inf` entries to
exactly 0 probability. But the thickness path does:

```python
conditioned_thick_logits = (slot_choice.unsqueeze(-1) * layer_logits).sum(dim=0)
```

For a padded row `p`: `slot_choice[p] = 0.0`, `layer_logits[p, :] = -inf`.
Under IEEE-754, `0.0 * -inf = NaN`. That NaN contaminates every bin
after `.sum(dim=0)`, poisons the downstream softmax, and yields a NaN
one-hot from the STE. Result: NaN thickness scalar → NaN Lab → NaN loss
→ every step skipped.

**Fix:**

```python
logits_step = torch.nan_to_num(logits_step, neginf=-1e9, posinf=1e9)
```

`-1e9` still gives `≈0` under softmax (relative to any finite logit),
but it's a real finite number, so multiplication is safe.

**Moral:** whenever you mix masks-as-`-inf` with multiplicative ops,
sanitize first. This is the same class of bug as `attention_mask * 0`
producing NaN with float32.

---

## The full gradient trace, one step at a time

Setting the scene: batch of B examples. Model forward runs *once* per
batch. Then a loop over `(b, k)` runs the sim + loss inside the rollout.

```
target_lab, pool_features, gt_slots, gt_thicknesses            (given)
                       │
                       ▼
        logits = model(target_lab, pool, structure_matrix_gt)
                       │  shape [B, MAX_LAYERS+1, 3201]
                       │
       (loop b in batch, k in 0..N_b-1)
                       │
                       ▼
          logits[b, k] ─── nan_to_num (Trick #4)
                       │
                       ▼
                 ste_pick (Trick #2)
                       │
        ┌──────────────┴──────────────┐
        ▼                             ▼
  slot_choice_k [32]           thickness_nm_k  [scalar]
   (hard 1-hot / STE grad)      (bin center / STE grad)
        │                             │
        │  n_layer_k = slot_choice_k @ pool_n
        │  k_layer_k = slot_choice_k @ pool_k
        ▼                             ▼
  n_layer_k, k_layer_k  [NUM_LAMBDA]  │
                       │              │
                       └──────┬───────┘
                              ▼
              full stack = detached GT prefix
                         + differentiable layer_k
                         + detached GT suffix          (Trick #3)
                              │
                              ▼
            predicted_lab = differentiable_sim(...)    (Trick #1)
                              │
                              ▼
             loss_bk = ciede2000_torch(target, pred)
                              │
                              ▼
                       accumulate over (b, k)
                              │
                              ▼
                        mean → .backward()
```

**Backward, arrow by arrow:**

```
grad(loss)                                             (= 1.0)
  │  ciede2000_torch is pure torch — normal autograd
  ▼
grad w.r.t. predicted_lab                              [3]
  │  DifferentiableStackSim.backward runs the JAX vjp  (Trick #1)
  ▼
grad w.r.t. n_layer_k   [NUM_LAMBDA]
grad w.r.t. k_layer_k   [NUM_LAMBDA]
grad w.r.t. thickness_nm_k  [scalar]
(prefix / suffix are detached — nothing goes back through them)  (Trick #3)
  │  n_layer_k = slot_choice_k @ pool_n is a plain linear op
  │  thickness_nm = thick_choice @ bin_centers is a plain linear op
  ▼
grad w.r.t. slot_choice_k   [32]
    ≈ ⟨grad_n, pool_n[i]⟩ + ⟨grad_k, pool_k[i]⟩  per slot i
grad w.r.t. thick_choice    [100]
    ≈ grad_thickness · bin_centers[j]            per bin j
  │  STE says: d(one_hot + probs - probs.detach()) / d(logits)
  │           = d(probs)/d(logits)                  (Trick #2)
  ▼
grad w.r.t. slot_probs, thick_probs
  │  softmax is a plain torch op
  ▼
grad w.r.t. logits[b, k, :]  [3201]
  │  transformer computed these logits from:
  │    * cross-attention over the pool encoder
  │    * causal self-attention over positions 0..k-1
  │    * per-slot thickness/eos heads
  │  All of those use SHARED parameters, so gradient reaches everything.
  ▼
grad w.r.t. every model parameter
```

Then contributions accumulate over all `(b, k)` in the batch. Every
parameter gets updates weighted by how much it contributed to producing
each of those position-k logits — which is "for every position in every
example." Plenty of signal.

---

## The uncomfortable-but-true bit

Even though the sim runs only ONCE per `(b, k)` with the argmax combo,
gradient reaches ALL 3201 vocab entries at that position — because the
sim's analytic gradient is *projected* onto every material and every
thickness bin via the linear pool-and-bin-center maps. That's the
whole reason this is tractable at 1M example scale: 1 sim call per
position, not 3201.

The caveat: this projection is a **first-order linear approximation**.
For alternate materials whose n/k is very different from the current
argmax, the projection can point the wrong direction (classic STE bias).
The small learning rates (5e-6 for Experiment A, 1e-6 for B) and the
starting point (a well-pretrained checkpoint) keep us in the neighborhood
where the projection is locally honest.

---

## Self-test — try answering these before scrolling

1. What operations run **once per batch** vs. **once per (example, position)**?
2. At position k, exactly which tensor in the sim's input carries
   `requires_grad=True`?
3. What three gradient tensors does `jax.vjp` hand back to torch when
   the sim's backward fires?
4. What would break if we forgot `.detach()` on the GT prefix / suffix
   in the full stack?
5. What would break if we skipped the `nan_to_num` in `ste_pick`?
6. Why do we STE the slot pick instead of just using a soft mixture
   `soft_probs @ pool_n`?
7. What's the "1 sim, 3201 gradients" claim, and where in the code does
   the projection happen?

<details>
<summary>Answers</summary>

1. **Once per batch:** the transformer forward. **Once per (b, k):**
   `ste_pick`, the sim call, `ciede2000_torch`. Model is called once;
   loss is evaluated N times per example.

2. Layer k's `n_layer_k`, `k_layer_k`, and `thickness_nm_k`. The
   prefix and suffix in the concatenation are `.detach()`'d.

3. `grad_n_stack`, `grad_k_stack`, `grad_thicknesses_nm`. `grad_incidence_angle`
   is returned as `None` because we don't differentiate through it.

4. Prefix/suffix are teacher-forced GT constants; if their gradients
   flowed we'd be trying to update the model based on GT values that
   didn't originate from the model. Silently corrupts the credit
   assignment. Loss would still be finite, so you'd only notice via
   worse convergence.

5. `0.0 * -inf = NaN` inside the thickness STE mixing. Every batch
   would produce NaN loss and every step would be skipped. This was
   the actual bug we hit and fixed.

6. Two reasons: (a) mixing n/k of physically distinct materials is
   not a real material — its optical response is nonsense; (b) even
   ignoring physics, "40% titanium + 60% chromium" is not what a
   fabrication process can build, so the sim would be simulating a
   design we can't manufacture. STE keeps the sim honest.

7. `jax.vjp` gives 3 gradient tensors at the argmax combo (∂ΔE/∂n_k,
   ∂ΔE/∂k_k, ∂ΔE/∂thickness_k). Those are pushed back through the
   linear maps `n_layer_k = slot_choice @ pool_n` and
   `thickness_nm = thick_choice @ bin_centers`, which (by the chain
   rule for a linear op) project the sim gradients onto every slot's
   spectrum and every bin's position. Happens implicitly via
   torch autograd on those two matmuls in `_rollout_one`.

</details>

---

## Instructions for a teaching agent

If you're an agent handed this document:

1. Read this file plus `analyses/de_finetune/README.md`,
   `src/de_finetune.py`, `src/optical_sim_diff.py`. Skim
   `src/model.py` for the cross-attn head interface.
2. Do NOT dump content back at the learner. Ask ONE question at a time.
3. Suggested sequence:
   - "In one sentence, what problem does this finetune stage solve?"
   - "What does the model output — shape and semantics?"
   - "Why can't we backprop straight through the model's argmax?"
   - "How does the STE trick let gradient flow through a hard pick?"
   - "Why do we sim the *full* stack instead of just the prefix?"
   - "Why do we `.detach()` the GT prefix and suffix?"
   - "What does `jax.vjp` return, and how does `DifferentiableStackSim.backward`
     use it?"
   - "What was the `nan_to_num` bug — what was multiplied by what?"
   - "The sim runs once per position, but every vocab entry gets a
     gradient. How?"
   - Final: "Trace one backward pass in your own words, from `loss.backward()`
     down to a specific model parameter."
4. Correct wrong mental models before moving on. Skip a topic if the
   learner clearly gets it. Save your longer explanations for when the
   learner explicitly asks or gets stuck twice on the same idea.
5. End by asking the learner to name the four tricks and what problem
   each solves.

To launch this pass in a fresh session:

```
Agent: general-purpose
Prompt: Please act as a teaching agent. Read
        analyses/de_finetune/GRADIENT_FLOW_EXPLAINER.md and follow the
        "Instructions for a teaching agent" section at the bottom.
        My starting knowledge: I understand transformers, backprop, and
        INDIGO's pretraining pipeline at a high level. I want to end
        this session able to trace a full backward pass and explain
        each of the four tricks.
```
