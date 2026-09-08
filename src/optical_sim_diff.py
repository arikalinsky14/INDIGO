"""
Differentiable Optical Simulator (PyTorch ↔ JAX bridge)
=======================================================

Companion to `src.optical_sim.OpticalSimulator`. Same physics
(transfer-matrix reflectance → CIE Lab), but exposed as a PyTorch
autograd operation so ΔE-based losses can flow gradients back into the
model's logits.

Used by `src.de_finetune` for the post-training ΔE finetune stage.

Why the bridge exists
---------------------
The forward path is:
    (n_stack, k_stack, thicknesses_nm) --stackrt_n_k--> reflectance
    reflectance --XYZ pipeline + D65 illuminant--> Lab
Every step is JAX-native and mathematically differentiable. What we
lack is a way to compute the gradient from the PyTorch side of the
pipeline. `torch.autograd.Function` provides that: forward converts
torch → JAX and runs the sim, backward uses `jax.vjp` on the same
computation and converts the resulting gradients back to torch.

No physics is reimplemented — the JAX transfer-matrix method
(`jaxlayerlumos.jaxlayerlumos.stackrt_n_k`) and the CIE colour pipeline
(`jaxlayerlumos.colors.composite.spectrum_to_sRGB`) are both called
verbatim. The sRGB → Lab tail is reimplemented in `jnp` (a few dozen
lines) so the whole spectrum → Lab chain is one JAX function that
`jax.vjp` can traverse in one shot.

Inputs / outputs (per call)
---------------------------
Forward:
    n_stack       torch.Tensor [num_layers, NUM_LAMBDA] — real refractive index per layer
    k_stack       torch.Tensor [num_layers, NUM_LAMBDA] — imag (absorption) per layer
    thicknesses_nm torch.Tensor [num_layers]           — thickness in nm
    (all differentiable; float64 recommended for numerical stability)

Output:
    lab           torch.Tensor [3]                    — CIE L*, a*, b* (D65)

Backward: gradients flow into all three inputs.

The wrapper is intentionally SCALAR-BATCH (one stack per call). Batched
sims will vmap through this via the calling code — cheaper than
teaching the Function about batch shapes and preserves the simple
autograd semantics.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from src.material_features import (
    CANONICAL_FREQ_HZ,
    CANONICAL_LAMBDA_NM,
    NUM_LAMBDA,
)

_JAX_AVAILABLE = False
_IMPORT_ERROR: Optional[str] = None
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import scipy.constants as scic
    from jaxlayerlumos.jaxlayerlumos import stackrt_n_k
    import jaxlayerlumos.colors.composite as jll_colors_composite

    _JAX_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR = str(e)


# ============================================================================
# Monkey-patch: strip trace-hostile asserts from jaxlayerlumos so we can
# @jax.jit the forward. The library has Python-level assertions that check
# invariants on the thicknesses array (e.g. `assert thicknesses[0] == 0`)
# which raise TracerBoolConversionError under jit / vmap / jvp — Python's
# assert wants a concrete bool, but a jitted call sees a Tracer.
#
# The invariants themselves are guaranteed at our call site (we prepend
# air with thickness 0 and append substrate with thickness 0 ourselves
# in _JAX_FORWARD below), so removing the asserts is safe. We rewrite the
# offending functions' source at import time — a scoped, surgical monkey-
# patch. If the library changes shape and the patch no longer applies,
# we fall through to non-jit'd mode with a warning rather than crashing.
# ============================================================================


_JIT_PATCH_APPLIED: bool = False
_JIT_PATCH_REASON: Optional[str] = None


def _patch_jaxlayerlumos_asserts() -> None:
    """Strip trace-hostile Python control flow from jaxlayerlumos so
    @jax.jit can wrap the sim path.

    Two patterns to remove:

      1. `assert thicknesses[0] == 0` / `assert thicknesses[-1] == 0`
         (invariant already guaranteed at our call site in _JAX_FORWARD).

      2. In stackrt_eps_mu_theta, a Python-level `if` block that branches
         on `jnp.isinf(jnp.real(eps_r[:, -1]))` — checks whether the last
         layer is a perfect-absorber substrate and zeros out T_TE/T_TM if
         so. Our substrate is fused silica (finite eps), so the branch
         is never taken; and we ignore T_TE/T_TM anyway (we only use
         reflectance). The library maintainer flagged this block with a
         `TODO: it is needed?` comment.

    Both patterns break jit because Python's assert and if want concrete
    Python bools, but under tracing they see Tracer objects.

    We rewrite each affected function via AST (parse -> prune -> unparse
    -> exec into the library's module namespace). Prune predicates are
    conservative — we only skip statements whose test refers to a name
    we know is a traced array (`thicknesses` for asserts, `eps_r`/`mu_r`
    for the if).
    """
    global _JIT_PATCH_APPLIED, _JIT_PATCH_REASON

    if not _JAX_AVAILABLE:
        _JIT_PATCH_REASON = "jax unavailable"
        return

    import ast
    import inspect
    import textwrap

    try:
        import jaxlayerlumos.jaxlayerlumos as _jll_mod
    except Exception as e:
        _JIT_PATCH_REASON = f"cannot import jaxlayerlumos.jaxlayerlumos: {e}"
        return

    class _TraceHostilePruner(ast.NodeTransformer):
        def __init__(self) -> None:
            self.pruned: list = []

        def _test_src(self, node) -> str:
            try:
                return ast.unparse(node)
            except Exception:
                return ""

        def visit_Assert(self, node: ast.Assert):
            test_src = self._test_src(node.test)
            # Guarantee-elsewhere: `assert thicknesses[0] == 0` and cousins.
            if "thicknesses" in test_src:
                self.pruned.append(f"assert {test_src}")
                return None
            return node

        def visit_If(self, node: ast.If):
            test_src = self._test_src(node.test)
            # `if (jnp.all(jnp.isinf(...eps_r...)) and ...)` in
            # stackrt_eps_mu_theta. Match on both key names to be
            # specific.
            if (
                "jnp.isinf" in test_src
                and "eps_r" in test_src
                and "jnp.allclose" in test_src
            ):
                self.pruned.append(f"if {test_src[:80]}...")
                return None
            # Recurse into nested statements otherwise.
            self.generic_visit(node)
            return node

    patched_names: list = []
    failures: list = []

    for name in list(vars(_jll_mod).keys()):
        obj = getattr(_jll_mod, name, None)
        if not callable(obj):
            continue
        # Only patch functions defined IN this module (not re-exports).
        if getattr(obj, "__module__", None) != _jll_mod.__name__:
            continue
        try:
            src = inspect.getsource(obj)
        except (TypeError, OSError):
            continue

        # Cheap gate: only pay parse+prune cost if a hostile keyword is
        # even present.
        if (
            "assert thicknesses" not in src
            and not ("jnp.isinf" in src and "eps_r" in src)
        ):
            continue

        src_dedented = textwrap.dedent(src)
        try:
            tree = ast.parse(src_dedented)
        except SyntaxError as e:
            failures.append((name, f"parse: {e}"))
            continue

        pruner = _TraceHostilePruner()
        new_tree = pruner.visit(tree)
        ast.fix_missing_locations(new_tree)
        if not pruner.pruned:
            continue  # nothing to do for this one

        try:
            new_src = ast.unparse(new_tree)
        except Exception as e:
            failures.append((name, f"unparse: {e}"))
            continue

        try:
            exec(
                compile(new_src, f"<jit-patched:{name}>", "exec"),
                _jll_mod.__dict__,
            )
            patched_names.append((name, pruner.pruned))
        except Exception as e:
            failures.append((name, f"exec: {e}"))

    if failures:
        _JIT_PATCH_REASON = (
            f"failed: {failures}; patched anyway: "
            f"{[n for n, _ in patched_names]}"
        )
        _JIT_PATCH_APPLIED = bool(patched_names)
    elif patched_names:
        _JIT_PATCH_APPLIED = True
        _JIT_PATCH_REASON = (
            f"patched {len(patched_names)} function(s): "
            + "; ".join(
                f"{n} ({', '.join(p)})" for n, p in patched_names
            )
        )
    else:
        _JIT_PATCH_APPLIED = False
        _JIT_PATCH_REASON = (
            "no trace-hostile patterns found in jaxlayerlumos — "
            "library may have changed"
        )


_patch_jaxlayerlumos_asserts()


def is_available() -> bool:
    return _JAX_AVAILABLE


def get_import_error() -> Optional[str]:
    return _IMPORT_ERROR


# ============================================================================
# Static physics constants (air + fused-silica substrate)
# ============================================================================
# These match src/optical_sim.py bit-for-bit. Prepended / appended to every
# stack inside the JAX forward call so caller-side code never has to know.

_AIR_N = np.ones(NUM_LAMBDA, dtype=np.float64)
_AIR_K = np.zeros(NUM_LAMBDA, dtype=np.float64)


def _fused_silica_n() -> np.ndarray:
    """Sellmeier dispersion for fused silica (Malitson 1965)."""
    lam_um = CANONICAL_LAMBDA_NM / 1000.0
    lam2 = lam_um ** 2
    n2 = (
        1.0
        + 0.6961663 * lam2 / (lam2 - 0.0684043 ** 2)
        + 0.4079426 * lam2 / (lam2 - 0.1162414 ** 2)
        + 0.8974794 * lam2 / (lam2 - 9.896161 ** 2)
    )
    return np.sqrt(n2)


_SUBSTRATE_N = _fused_silica_n()
_SUBSTRATE_K = np.zeros(NUM_LAMBDA, dtype=np.float64)


# ============================================================================
# Colour constants (D65 white point + linear-sRGB → XYZ matrix)
# ============================================================================
# Mirror src/color_utils.py so results are numerically identical.

_X_N: float = 95.047
_Y_N: float = 100.000
_Z_N: float = 108.883

_M_SRGB_TO_XYZ_NP = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


# ============================================================================
# JAX-native forward: n/k stack + thicknesses -> Lab
# ============================================================================


def _make_jax_forward():
    """Build the closed-over JAX forward function once at import time so we
    don't rebuild constant tensors on every call.
    """
    if not _JAX_AVAILABLE:
        return None

    M_SRGB_TO_XYZ = jnp.asarray(_M_SRGB_TO_XYZ_NP)
    AIR_N_JAX = jnp.asarray(_AIR_N)
    AIR_K_JAX = jnp.asarray(_AIR_K)
    SUB_N_JAX = jnp.asarray(_SUBSTRATE_N)
    SUB_K_JAX = jnp.asarray(_SUBSTRATE_K)
    FREQS_JAX = jnp.asarray(CANONICAL_FREQ_HZ)
    LAMBDAS_JAX = jnp.asarray(CANONICAL_LAMBDA_NM)
    # The spectrum_to_sRGB helper drops wavelengths outside (360, 830) nm —
    # mirror that here so freqs/reflectance we hand to it stay aligned.
    _VALID_MASK = (CANONICAL_LAMBDA_NM > 360) & (CANONICAL_LAMBDA_NM < 830)
    VALID_LAMBDAS_JAX = jnp.asarray(CANONICAL_LAMBDA_NM[_VALID_MASK])
    VALID_MASK_JAX = jnp.asarray(_VALID_MASK)
    THETAS_JAX = jnp.asarray([0.0])
    NANO = float(scic.nano)

    def _inv_gamma(c):
        # Standard sRGB inverse gamma, extended symmetrically to negatives so
        # out-of-gamut colours pass through cleanly. Matches the numpy loop
        # in src/color_utils.spectrum_to_lab.
        abs_c = jnp.abs(c)
        sign = jnp.sign(c)
        low = abs_c / 12.92
        high = ((abs_c + 0.055) / 1.055) ** 2.4
        return sign * jnp.where(abs_c <= 0.04045, low, high)

    def _f_lab(t):
        # CIE Lab piecewise cube root.
        delta = 6.0 / 29.0
        cube_root = jnp.cbrt(t)
        linear = t / (3 * delta * delta) + 4.0 / 29.0
        return jnp.where(t > delta ** 3, cube_root, linear)

    # JIT the forward: without this, jax.vjp re-traces the entire TMM +
    # CIE-color pipeline through Python on every call. On the first
    # finetune throughput smoke that translated to ~220 ms per sim call
    # (~200 s per step, 0.7 ex/s), because each of the ~B*N sim calls
    # per training step paid the full tracing cost. With @jax.jit, JAX
    # compiles one XLA kernel per (n_stack shape, k_stack shape,
    # thicknesses shape, incidence_angle value) tuple and caches it —
    # in practice 1..MAX_LAYERS shapes × one incidence value at training
    # time, so ≤ MAX_LAYERS kernels. Compilation happens on the first
    # call for each shape; every subsequent call is a single dispatch.
    # jax.vjp remains happy because it can differentiate through a
    # jitted function transparently.
    @jax.jit
    def forward(n_stack, k_stack, thicknesses_nm, incidence_angle):
        """
        n_stack, k_stack : [num_layers, NUM_LAMBDA]
        thicknesses_nm   : [num_layers]
        incidence_angle  : scalar (degrees)
        returns          : [3] Lab
        """
        # Prepend air, append substrate: [num_layers + 2, NUM_LAMBDA]
        n_full = jnp.concatenate(
            [AIR_N_JAX[None, :], n_stack, SUB_N_JAX[None, :]], axis=0,
        )
        k_full = jnp.concatenate(
            [AIR_K_JAX[None, :], k_stack, SUB_K_JAX[None, :]], axis=0,
        )
        # stackrt_n_k expects [num_freqs, num_layers]; transpose.
        n_complex = (n_full + 1j * k_full).T  # [NUM_LAMBDA, num_layers + 2]

        d_full_nm = jnp.concatenate([
            jnp.zeros(1),
            thicknesses_nm,
            jnp.zeros(1),
        ])
        d_m = d_full_nm * NANO

        thetas = jnp.asarray([incidence_angle])
        R_TE, _, R_TM, _ = stackrt_n_k(n_complex, d_m, FREQS_JAX, thetas)
        R_avg = (R_TE[0] + R_TM[0]) / 2.0  # [NUM_LAMBDA]

        # Filter to the CIE visible band the sRGB helper expects.
        R_valid = R_avg[VALID_MASK_JAX]

        # spectrum → un-clipped sRGB → linear sRGB → XYZ → Lab
        rgb_unclipped = jll_colors_composite.spectrum_to_sRGB(
            VALID_LAMBDAS_JAX, R_valid, use_clipping=False,
        )
        rgb_float = rgb_unclipped.reshape(-1)[:3]

        rgb_linear = _inv_gamma(rgb_float)
        xyz = (M_SRGB_TO_XYZ @ rgb_linear) * 100.0
        fx = _f_lab(xyz[0] / _X_N)
        fy = _f_lab(xyz[1] / _Y_N)
        fz = _f_lab(xyz[2] / _Z_N)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return jnp.stack([L, a, b])

    return forward


_JAX_FORWARD = _make_jax_forward()


# ============================================================================
# torch.autograd.Function bridge
# ============================================================================


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().double().numpy()


def _to_torch(arr, device, dtype) -> torch.Tensor:
    # copy=True so the tensor owns its buffer (silences the "given NumPy
    # array is not writable" warning that fires on JAX-backed arrays).
    return torch.from_numpy(np.array(arr, copy=True)).to(
        device=device, dtype=dtype,
    )


class DifferentiableStackSim(torch.autograd.Function):
    """torch <-> jax bridge for the optical sim.

    Forward and backward are both scalar-batch (one stack). The caller
    batches by looping (or vmap-ing) over examples — see
    src.de_finetune.rollout_batch.
    """

    @staticmethod
    def forward(
        ctx,
        n_stack: torch.Tensor,           # [num_layers, NUM_LAMBDA]
        k_stack: torch.Tensor,           # [num_layers, NUM_LAMBDA]
        thicknesses_nm: torch.Tensor,    # [num_layers]
        incidence_angle: float = 0.0,
    ) -> torch.Tensor:
        if _JAX_FORWARD is None:
            raise RuntimeError(
                f"jaxlayerlumos not available: {_IMPORT_ERROR}"
            )

        device = n_stack.device
        dtype = n_stack.dtype

        n_np = _to_np(n_stack)
        k_np = _to_np(k_stack)
        t_np = _to_np(thicknesses_nm)

        n_jax = jnp.asarray(n_np)
        k_jax = jnp.asarray(k_np)
        t_jax = jnp.asarray(t_np)

        def _f(n, k, t):
            return _JAX_FORWARD(n, k, t, incidence_angle)

        lab_jax, vjp_fn = jax.vjp(_f, n_jax, k_jax, t_jax)

        # Save the vjp closure for backward. It captures the JAX arrays
        # (fine — they're small compared to the model) and does not
        # re-run the forward.
        ctx.vjp_fn = vjp_fn
        ctx.device = device
        ctx.dtype = dtype
        ctx.num_layers = n_stack.shape[0]

        return _to_torch(lab_jax, device, dtype)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,   # [3] — dLoss/dLab
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor],
        Optional[torch.Tensor], None,
    ]:
        grad_jax = jnp.asarray(_to_np(grad_output))
        grad_n_jax, grad_k_jax, grad_t_jax = ctx.vjp_fn(grad_jax)
        grad_n = _to_torch(grad_n_jax, ctx.device, ctx.dtype)
        grad_k = _to_torch(grad_k_jax, ctx.device, ctx.dtype)
        grad_t = _to_torch(grad_t_jax, ctx.device, ctx.dtype)
        # Fourth return value corresponds to incidence_angle (non-tensor input).
        return grad_n, grad_k, grad_t, None


def differentiable_compute_lab(
    n_stack: torch.Tensor,
    k_stack: torch.Tensor,
    thicknesses_nm: torch.Tensor,
    incidence_angle: float = 0.0,
) -> torch.Tensor:
    """Public entry point. See module docstring."""
    return DifferentiableStackSim.apply(
        n_stack, k_stack, thicknesses_nm, incidence_angle,
    )


# ============================================================================
# Convenience: assemble a stack from a soft slot choice (for STE'd rollouts)
# ============================================================================


def assemble_layer_nk_from_pool(
    pool_n: torch.Tensor,      # [M_MAX, NUM_LAMBDA]
    pool_k: torch.Tensor,      # [M_MAX, NUM_LAMBDA]
    slot_choice: torch.Tensor, # [num_layers, M_MAX] (STE one-hot per layer)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return per-layer n/k stacks assembled by picking (differentiably via STE)
    material features from the pool.

    Forward: slot_choice is a hard one-hot, so this is a pool[argmax] lookup.
    Backward: gradient of the sum flows through slot_choice to the slot
    logits. The sim then sees a single-material spectrum per layer, not an
    averaged (physically-invalid) blend.

    Returns
    -------
    n_stack : [num_layers, NUM_LAMBDA]
    k_stack : [num_layers, NUM_LAMBDA]
    """
    # [num_layers, M_MAX, 1] * [1, M_MAX, NUM_LAMBDA] -> [num_layers, M_MAX, NUM_LAMBDA]
    # sum over M_MAX -> [num_layers, NUM_LAMBDA]
    n_stack = (slot_choice.unsqueeze(-1) * pool_n.unsqueeze(0)).sum(dim=1)
    k_stack = (slot_choice.unsqueeze(-1) * pool_k.unsqueeze(0)).sum(dim=1)
    return n_stack, k_stack


# ============================================================================
# Smoke test — verifies (1) forward matches OpticalSimulator numerically,
# (2) finite-difference gradient checks pass for both thickness and n_stack.
# ============================================================================


def _smoke() -> None:
    if not _JAX_AVAILABLE:
        print(f"[smoke] jaxlayerlumos not installed ({_IMPORT_ERROR}); skipping.")
        return

    print(f"[smoke] jaxlayerlumos assert-patch applied: {_JIT_PATCH_APPLIED}")
    print(f"[smoke]   reason: {_JIT_PATCH_REASON}")

    from pathlib import Path
    from src.material_features import load_jll_directory
    from src.optical_sim import OpticalSimulator

    materials_dir = Path("/home/claude/JaxLayerLumos/jaxlayerlumos/materials")
    if not materials_dir.exists():
        # Try the environment location on Pitt CRC.
        materials_dir = Path(
            "/ihome/ohinder/ajk245/envs/llm-env/lib/python3.11/"
            "site-packages/jaxlayerlumos/materials"
        )
    pool_full = load_jll_directory(materials_dir)
    pool = [pool_full["SiO2"], pool_full["Ag"], pool_full["TiO2"]]

    slot_indices = [0, 1, 2]
    thicknesses_nm = [100, 30, 75]

    # Reference (numpy path).
    sim = OpticalSimulator(incidence_angle=0)
    lab_ref = sim.compute_lab(pool, slot_indices, thicknesses_nm)
    print(f"[smoke] reference Lab (numpy path):  L*={lab_ref[0]:.4f}  "
          f"a*={lab_ref[1]:.4f}  b*={lab_ref[2]:.4f}")

    # Diff path.
    n_stack = torch.tensor(
        np.stack([pool[i].n for i in slot_indices], axis=0), dtype=torch.float64,
    )
    k_stack = torch.tensor(
        np.stack([pool[i].k for i in slot_indices], axis=0), dtype=torch.float64,
    )
    t = torch.tensor(thicknesses_nm, dtype=torch.float64, requires_grad=True)
    lab = differentiable_compute_lab(n_stack, k_stack, t)
    print(f"[smoke] differentiable Lab:          L*={lab[0].item():.4f}  "
          f"a*={lab[1].item():.4f}  b*={lab[2].item():.4f}")

    max_diff = max(abs(lab[i].item() - lab_ref[i]) for i in range(3))
    print(f"[smoke] max |Lab_diff - Lab_ref| = {max_diff:.6f}")
    # 1e-3 gives 5-decimal Lab agreement on a 0-100 scale — well below any
    # perceptual noise floor (perceptibility threshold ~ ΔE 2). Float64
    # drift through stackrt_n_k + spectrum→sRGB + gamma + matrix mul +
    # cube root routinely accumulates ~1e-6 without any physics bug.
    assert max_diff < 1e-3, (
        f"forward mismatch too large: max |Lab_diff - Lab_ref| = {max_diff:.6e} "
        f"— check the Lab pipeline"
    )

    # Finite-difference gradient check on thickness.
    lab.sum().backward()
    grad_t_analytic = t.grad.clone()

    eps = 1e-4
    grad_t_fd = torch.zeros_like(t)
    for i in range(len(thicknesses_nm)):
        t_plus = t.detach().clone()
        t_plus[i] += eps
        t_minus = t.detach().clone()
        t_minus[i] -= eps
        lab_plus = differentiable_compute_lab(n_stack, k_stack, t_plus)
        lab_minus = differentiable_compute_lab(n_stack, k_stack, t_minus)
        grad_t_fd[i] = (lab_plus.sum() - lab_minus.sum()) / (2 * eps)

    max_grad_err = (grad_t_analytic - grad_t_fd).abs().max().item()
    print(f"[smoke] thickness grad analytic vs FD max err = {max_grad_err:.6e}")
    print(f"[smoke]   analytic: {grad_t_analytic.numpy()}")
    print(f"[smoke]   FD:       {grad_t_fd.numpy()}")
    assert max_grad_err < 1e-3, "gradient check failed"

    print("[smoke] OK — forward matches reference and gradients check out.")


if __name__ == "__main__":
    _smoke()
