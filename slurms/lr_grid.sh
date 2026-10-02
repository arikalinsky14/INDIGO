#!/usr/bin/env bash
#SBATCH --job-name=indigo-lr-grid
#SBATCH --output=job-outputs/indigo-lr-grid.%A_%a.out
#SBATCH --error=job-outputs/indigo-lr-grid.%A_%a.err

#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --time=06:00:00
#SBATCH --qos=short
#SBATCH --mail-user=ajk245@pitt.edu
#SBATCH --mail-type=END,FAIL,TIME_LIMIT

set -euo pipefail

# ============================================================================
# Per-configuration optimizer tuning, Porian et al. style
# ============================================================================
#
# They tune EXHAUSTIVELY at every configuration up to some scale and fit the
# laws to those optima. INDIGO's deployed law rests on three measurements, none
# of which bracketed its own optimum. This replaces that.
#
# THREE axes, not four. Batch size is NOT swept: on INDIGO it is a VRAM
# decision, the largest batch the card holds, because these models are
# input-bound rather than GPU-bound. src/scaling/configs.py:DEFAULT_BATCH_SIZE
# is the single source of truth and this script reads it, so the tuning batch
# size and the sweep batch size cannot drift apart. They must match: the
# optimal learning rate depends on batch size, so a law tuned at one and
# applied at another is measuring the wrong thing.
#
#   N      model size, taken from the sweep's own ladder so the tuned configs
#          ARE the configs it trains.
#   D      dataset size via --limit-examples at EPOCHS=1. The deployed law has
#          no D term while the sweep spans 9x in D at fixed N.
#   beta2  AdamW. Never tuned on INDIGO, which has always run torch's 0.999,
#          the top of the {0.95, 0.99, 0.999} they sweep. They report tuning it
#          is essential at LOW batch size, which is the regime we train in.
#
# D is held fixed in EXAMPLES, so every cell sees the same data regardless of
# anything else. That matches their design: within a model size their token
# budget is constant to within 1% across every cell.
#
# ---------------------------------------------------------------------------
# NOTHING IS EVER OVERWRITTEN
# ---------------------------------------------------------------------------
# Each cell writes one file named for (epochs, D, d_model, slot layers, batch
# size, beta2). lr_tuning.py checks for it BEFORE loading data or touching the
# GPU and refuses to replace it. This script passes --skip-existing, so
# re-submitting an array is free and finished cells are left alone. Pass
# FORCE=1 only when you mean to redo a cell.
#
# ---------------------------------------------------------------------------
# Stages, and why stage 2 is not a fixed-D ladder
# ---------------------------------------------------------------------------
#
# Porian et al. tune at a CONSTANT token multiplier: across a 42x range in
# parameters their sweep holds M = tokens/params between 20.0 and 21.1. N and M
# are decoupled by construction, so a law in N alone is the right object for
# them, and their compute-optimal M is itself ~constant, so one M covers the
# whole study.
#
# INDIGO cannot borrow that. Our compute-optimal multiplier is NOT constant:
# D*/N* runs 29.3 down to 0.86 across our budgets, a 34x range, and the 48
# sweep runs occupy M from 0.28 to 52. Tuning at a fixed D would be worse than
# useless, because M = D/N would then vary as 1/N across the ladder, 220x: the
# fitted "lr(N)" would really be lr along a trajectory in M. That is the
# aspect-ratio confound again, in a different variable.
#
#   STAGE=probe  one cell, ONE learning rate.          1 cell,  <1 GPU-h
#            Measures examples per second. Two decisions ride on it.
#
#            FIRST, whether the rest is affordable at all. The first stage-1
#            submission ran at a median of 204 ex/s where the sweep reaches
#            2171 on the same shards, and which of those holds decides whether
#            everything below costs 20 GPU-hours or 200.
#
#            SECOND, HOW MANY IsoFLOP CURVES STAGE 2 TUNES IN FULL, with the
#            remaining rungs projected from the fitted law instead. Every
#            added rung improves the conditioning of the (N, M) fit and
#            roughly doubles the cost:
#
#              rungs  cells   GPU-h @2171   GPU-h @204   cond   sd(b)  sd(c)  extrap
#                  2     12          31.3        245.3    630   0.100  0.054   13/24
#                  3     18          56.7        471.3    401   0.050  0.031    9/24
#                  4     24          103.7        927.6    303   0.032  0.021    5/24
#
#            sd(b) and sd(c) are the spreads of the recovered N and M
#            exponents over 2000 synthetic draws at 0.10 of noise in log-lr
#            units. "extrap" counts the sweep's 24 distinct model sizes
#            falling ABOVE the largest size tuned, so the fourth curve buys
#            coverage as well as conditioning, and coverage is the stronger
#            argument: an extrapolated size is a size whose learning rate is
#            a guess. At 2171 ex/s it tightens the N exponent 1.6x and halves
#            the extrapolated sizes for twice the compute, which is a real
#            judgement call; at 204 ex/s three curves already costs 471
#            GPU-hours and four is out of reach, so the answer is three or
#            nothing. Carry it into stage 2 as RUNGS=.
#
#            Run it ALONE: --array=0-0 and nothing else of yours queued, or
#            it measures contention rather than throughput.
#
#   STAGE=1  beta2 at both ends of the sweep's ladder.  6 cells, ~15 GPU-h
#            Does beta2 matter, and does its optimum move with scale? Their
#            data says it tracks BATCH SIZE, which we hold fixed, so one value
#            should serve; this checks that on our model.
#
#            THREE learning rates per cell, not seven. beta2 and the learning
#            rate interact, so comparing beta2 at one fixed rate can pick
#            whichever beta2 suits that rate; three is enough to see whether
#            the beta2 RANKING is stable across them. Locating the optimum
#            itself is stage 2's job. The three sit at prior/4, prior and
#            prior*4 (LR_SPAN=4 for this stage, not the default 30), so none
#            is so far off that it only measures divergence.
#
#            At each rung's own D*, not the historical 614,400. The first
#            attempt used that smaller budget and nothing learned: accuracy sat
#            at the EOS base rate for every trial and DeltaE came back as
#            scatter from 25 to 37, so no beta2 could be ranked against
#            another. The sweep's own run at that model size needed D = 4.2M
#            to reach DeltaE 12.4.
#
#   STAGE=beta2  the defensible beta2 study.        45 cells, ~156 GPU-h
#            Stage 1 could not separate 0.95 from 0.99: one seed, and only
#            two stable learning rates per cell, so no beta2 had a bracketed
#            LR optimum. This runs beta2 in {0.9, 0.95, 0.98, 0.99, 0.999} at
#            three sizes (smallest, middle and largest usable rung's N*),
#            three seeds, and seven learning rates per cell in sqrt(2) steps
#            centred on the prior, with per-example DeltaE saved. Seed is the
#            outer loop: --array=0-14 is a complete single-seed pass (~52
#            GPU-h), 15-44 adds the replicates. scripts/fit_beta2.py applies a
#            decision rule fixed before the data and draws the figure.
#            Writes to outputs/lr_search/beta2/.
#
#   STAGE=2  EVERY model on the lowest RUNGS curves.  18 cells, ~57 GPU-h
#            (RUNGS=3 by default; the probe decides 3 vs 4, see above)
#            The LR search runs INSIDE the IsoFLOP test. Not one point per
#            rung: the whole curve, because the curve is what the parabola is
#            fitted through, and a point whose LR was extrapolated moves the
#            minimum as surely as one trained wrong.
#
#            Each point is its own (N, M), which is what makes the law
#            identifiable here. Within one rung C is fixed, so
#            log M = const - 2 log N and the two columns are collinear; a
#            second rung shifts the intercept and separates them. Three rungs
#            give a condition number of 401 and recover both exponents to
#            +/- 0.05 under 10% noise. So the multiplier axis comes free from
#            the geometry, with no fractional scoring and no constant LR
#            needed, which means the law is measured under the schedule the
#            sweep actually trains with.
#
#            This is the one place we knowingly diverge from Porian et al.,
#            and it is forced: they tune at a constant multiplier (20.0 to
#            21.1 while parameters vary 42x), so a law in N alone is right for
#            them. Ours cannot be, since D*/N* runs 29.3 to 0.86.
#
#   STAGE=check  the extrapolation check.               1 cell,  ~6 GPU-h
#            Run between 2 and 3, before stage 3 spends anything. Stage 3
#            trusts the fitted law at every upper-rung point, so tune the
#            compute-optimal point of the highest usable rung in full and
#            compare predicted against measured. If the ratio is far from 1,
#            the law does not reach: tune one more rung directly (RUNGS + 1)
#            instead of spending stage 3 on rates that are guesses. Writes to
#            outputs/lr_search/check/, so the law is never fitted on the point
#            that tests it.
#
#   STAGE=3  finish the IsoFLOP.                       30 cells, ~61 GPU-h
#            Every (N, D) point of the remaining curves, repeat seeds
#            included, trained ONCE at the learning rate the fitted law gives
#            for that point's own N and D. The LR is the only thing
#            extrapolated: beta2 is stage 1's winner and the grid is the
#            sweep's. Also the lower rungs' repeat seeds, at the rate stage 2
#            picked for their seed-42 twin, so every seed cluster sits at one
#            LR. Writes to outputs/isoflop_tuned/.
#
#            Stages 2 and 3 together ARE the final IsoFLOP figure: stage 2's
#            winning trial at each lower-rung point is that point's run.
#            scripts/collect_isoflop.py assembles them and refuses any point
#            not run like the rest (shard alignment, beta2, epochs, seed).
#
#            The default 30 includes the 2.5e16 rung (8 cells, ~28 GPU-h),
#            whose argmin sat on the boundary in the first sweep. Its largest
#            run is 58M example-passes, 1.45 epochs, ~8 h at 2171 ex/s.
#
# About 140 GPU-h in total at 2171 ex/s (probe 0.6, stage 1 15, stage 2 57,
# check 6, stage 3 61), startup and evals included, against the 566 runs
# behind their laws.
#
# Every stage passes --limit-shard-aligned, as the sweep's training.py runs
# always have. Earlier tuning cells did not: a 3.4M-example subset scattered
# over all 8,000 shards made each trial stream the entire 40M-row corpus, an
# I/O amplification of roughly 40M/D. That is the likeliest explanation for
# the first stage-1 submission's 204 ex/s against the sweep's 2171 (11.6x
# predicted at D = 3.45M, 10.6x observed), so the probe should now be re-run
# before anything is sized at 204.
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
#   STAGE=probe sbatch --array=0-0 --time=02:00:00 slurms/lr_grid.sh
#
#   # read the ex/s off the log. It sizes everything below AND decides how
#   # many curves stage 2 tunes, so price both before committing. --list works
#   # for every stage before it can run, stage 3 included:
#   STAGE=1 EXAMPLES_PER_SEC=<measured> bash slurms/lr_grid.sh --list
#   STAGE=2 RUNGS=3 EXAMPLES_PER_SEC=<measured> bash slurms/lr_grid.sh --list
#   STAGE=3 RUNGS=3 EXAMPLES_PER_SEC=<measured> bash slurms/lr_grid.sh --list
#
#   STAGE=1 sbatch --array=0-5%2 --time=<from the table> slurms/lr_grid.sh
#
#   # then, with BETA2_WINNER set to what stage 1 picked and RUNGS set to what
#   # the probe justified (3 -> --array=0-17, 4 -> --array=0-23):
#   STAGE=2 RUNGS=3 BETA2_WINNER=0.99 sbatch --array=0-17%2 \
#       --time=<from the table> slurms/lr_grid.sh
#   python scripts/fit_lr_law.py --results-dir outputs/lr_search/cross_attn \
#       --coverage-from analyses/scaling/results/isoflop_fit.json
#
#   STAGE=check BETA2_WINNER=0.99 sbatch --array=0-0 \
#       --time=<from the table> slurms/lr_grid.sh
#
#   # only if the check passes; RUNGS must be the value stage 2 ran with:
#   STAGE=3 RUNGS=3 BETA2_WINNER=0.99 sbatch --array=0-29%2 \
#       --time=<from the table> slurms/lr_grid.sh
#   python scripts/collect_isoflop.py --beta2 0.99 --rungs 3
#   python scripts/fit_scaling_porian.py \
#       --fit analyses/scaling/results/isoflop_tuned.json \
#       --output analyses/scaling/results/porian_fit_tuned.json
#
# Any cell whose optimum lands on a grid endpoint is DISCARDED by the fit, not
# averaged in. Widen LR_SPAN_DOWN or LR_SPAN_UP (whichever side the optimum
# fell on) for those cells and re-run them with FORCE=1.
# ============================================================================

# Environment FIRST. Everything below shells out to python, and the cell list
# in particular is computed before any task runs.
#
# Under sbatch this is the submit directory; run by hand it is wherever you
# already are, so --list works from a checkout without guessing a path.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    cd "${SLURM_SUBMIT_DIR}"
elif [[ ! -f scripts/lr_grid_cells.py && -d "$HOME/Github/INDIGO" ]]; then
    cd "$HOME/Github/INDIGO"
fi
if [[ ! -f scripts/lr_grid_cells.py ]]; then
    echo "run this from the INDIGO checkout (no scripts/lr_grid_cells.py here)" >&2
    exit 1
fi

# -------------------- Environment Setup --------------------
# Same two lines every other slurm in this repo uses. An earlier version of
# this script invented a conda activation that does not exist on this cluster
# and swallowed the failure with `|| true`, so the job ran against the system
# python and died on `import torch` after the scheduler had already given it a
# GPU. Failures here are fatal and loud.
if command -v module >/dev/null 2>&1; then
    module purge
    module load python/pytorch_251_311_cu124
fi
if [[ -f "$HOME/envs/llm-env/bin/activate" ]]; then
    source "$HOME/envs/llm-env/bin/activate"
fi
export TOKENIZERS_PARALLELISM=false

mkdir -p job-outputs

# Batch size comes from the sweep planner, never from a default here.
if [[ -z "${BATCH_SIZE:-}" ]]; then
    BATCH_SIZE="$(python3 -c 'import sys; sys.path.insert(0,".");
from src.scaling.configs import DEFAULT_BATCH_SIZE; print(DEFAULT_BATCH_SIZE)')" \
        || { echo "could not read DEFAULT_BATCH_SIZE. Is the environment "\
                  "active? Set CONDA_ENV, or pass BATCH_SIZE explicitly." >&2
             exit 1; }
fi

STAGE="${STAGE:-1}"
# The probe runs ONE learning rate; every other stage runs the full grid.
# Grid width is per stage: the probe times one run, stage 1 ranks beta2 across
# a small spread of learning rates, stage 2 locates the optimum.
#
# Stage 1's span is narrow on purpose. The grid is logspace(prior/SPAN,
# prior*SPAN), so at the default SPAN of 30 three points land at prior/30,
# prior and prior*30, a 900x spread. Two of the three would then sit far from
# the optimum, where beta2 mostly decides whether a run diverges rather than
# how well it trains, and the "ranking" would measure stability at absurd
# learning rates. At 4 the outer points are a factor of 4 either side of the
# prior: wide enough to see whether the ranking holds, close enough that every
# point is a plausible operating rate.
#
# Stage 2 and the check reach DOWN 8x and UP 5x from the prior, set from what
# stage 1 measured (job 4126638): at both ends of the ladder and under all
# three beta2, prior/4 lost to the prior by 0.8 to 2.6 DeltaE and 4x the prior
# DIVERGED, all six times. A symmetric 30x window would have put two of seven
# points past divergence and two more far below anything competitive, leaving
# prior/3.1, prior and 3.1x prior to locate the optimum: a three-point grid
# again. Seven points over prior/8 to 5x prior step 1.85x instead of 3.1x, and
# the top point is expected to diverge, which is what brackets from above.
case "${STAGE}" in
  probe)   N_LRS=1 ;;
  1)       N_LRS="${N_LRS:-3}"; LR_SPAN="${LR_SPAN:-4}" ;;
  # beta2 study: seven rates in sqrt(2) steps, prior/2.83 to 2.83 x prior,
  # centred so the prior itself is the middle point.
  beta2)   N_LRS="${N_LRS:-7}"
           LR_SPAN_DOWN="${LR_SPAN_DOWN:-2.828427}"
           LR_SPAN_UP="${LR_SPAN_UP:-2.828427}" ;;
  2|check) LR_SPAN_DOWN="${LR_SPAN_DOWN:-${LR_SPAN:-8}}"
           LR_SPAN_UP="${LR_SPAN_UP:-${LR_SPAN:-5}}" ;;
  3)       N_LRS=1 ;;               # one rate per point, given by the cell
esac
FIT="${FIT:-analyses/scaling/results/porian_fit.json}"
BETA2_WINNER="${BETA2_WINNER:-0.99}"
# How many of the lowest IsoFLOP curves stage 2 tunes in full. The rest are
# projected from the fitted law. The throughput probe decides 3 against 4: see
# the stage table at the top of this file.
RUNGS="${RUNGS:-3}"

LR_SPAN="${LR_SPAN:-30}"
# The grid runs prior/LR_SPAN_DOWN to prior*LR_SPAN_UP; both default to LR_SPAN.
LR_SPAN_DOWN="${LR_SPAN_DOWN:-${LR_SPAN}}"
LR_SPAN_UP="${LR_SPAN_UP:-${LR_SPAN}}"
N_LRS="${N_LRS:-7}"
SELECTION_METRIC="${SELECTION_METRIC:-delta_e}"
HEAD_MODE="${HEAD_MODE:-cross_attn}"
DATA_DIR="${DATA_DIR:-/ix1/ohinder/ajk245/Github/INDIGO/data/train}"
# Stages probe, 1 and 2 share the directory fit_lr_law.py reads. The check and
# stage 3 get their own: the check must not be fitted into the law it tests,
# and stage 3's single-rate runs are IsoFLOP points, not tuning evidence.
STAGE2_DIR="${STAGE2_DIR:-outputs/lr_search/${HEAD_MODE}}"
case "${STAGE}" in
  probe) OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/probe/${HEAD_MODE}}" ;;
  # Stage 1 ranks beta2 on three rates; three cannot bracket an LR optimum,
  # and the LR fit collapses beta2 by taking the best at each rate, so these
  # must not feed the law that stage 2 measures at one fixed beta2.
  1)     OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/stage1/${HEAD_MODE}}" ;;
  beta2) OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/beta2/${HEAD_MODE}}" ;;
  check) OUTPUT_DIR="${OUTPUT_DIR:-outputs/lr_search/check/${HEAD_MODE}}" ;;
  3)     OUTPUT_DIR="${OUTPUT_DIR:-outputs/isoflop_tuned/${HEAD_MODE}}" ;;
  *)     OUTPUT_DIR="${OUTPUT_DIR:-${STAGE2_DIR}}" ;;
esac
# MEASURED by the shard-aligned probe (job 4125957, d120/se4, D = 614,400):
# median 2156 ex/s over 24 step samples, harmonic mean 2033. The harmonic mean
# is the one that sizes wall time, since time per example is what adds up. The
# same probe without shard alignment (job 4125615) ran at ~340 and timed out,
# and the first stage-1 submission at 204: the old figure was I/O, not a floor.
EXAMPLES_PER_SEC="${EXAMPLES_PER_SEC:-2033}"

# Every stage trains on the schedule the SWEEP uses, cosine, because the law
# is applied to sweep runs and a learning rate means something different under
# a schedule whose shape depends on total steps.
#
# This is a change from an earlier design in which stage 3 got the multiplier
# axis by scoring one constant-LR run at fractions of its dataset. That trick
# is no longer needed: stage 2 tunes whole IsoFLOP curves, so every cell
# already carries its own (N, M) and the multiplier axis comes free from the
# geometry. Stage 3 is now the extrapolation CHECK, and it has to run under
# the sweep's own schedule or it is not checking the sweep's configuration.
# EVAL_FRACTIONS is still honoured if set, and lr_tuning.py refuses it unless
# LR_SCHEDULE=constant is set with it.
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"
EVAL_FRACTIONS="${EVAL_FRACTIONS-}"

CELL_ARGS=(--stage "${STAGE}" --fit "${FIT}" --beta2 "${BETA2_WINNER}"
           --rungs "${RUNGS}" --stage2-dir "${STAGE2_DIR}")

# --list prices the stage, so it must work BEFORE the stage can run: stage 3's
# learning rates do not exist until stage 2 is fitted, but its cells and cost
# do. It therefore asks for a table and a count, never the runnable lines.
if [[ "${1:-}" == "--list" ]]; then
    python3 scripts/lr_grid_cells.py "${CELL_ARGS[@]}" --format table \
        --n-lrs "${N_LRS}" --rate "${EXAMPLES_PER_SEC}" \
        --val-examples "${LIMIT_VAL_EXAMPLES:-10000}" \
        --de-examples "${LIMIT_DE_EXAMPLES:-2048}" \
        --wall-hours "${WALL_HOURS:-6}"
    N_TASKS="$(python3 scripts/lr_grid_cells.py "${CELL_ARGS[@]}" --format count)"
    echo
    echo "batch size ${BATCH_SIZE};  submit with --array=0-$((N_TASKS - 1))"
    exit 0
fi

# Cells come from scripts/lr_grid_cells.py, not from a ladder written here,
# because they are DERIVED from the sweep's own grid. See that script's
# docstring for why they have to be.
mapfile -t CELL_LINES < <(python3 scripts/lr_grid_cells.py "${CELL_ARGS[@]}")
N_TASKS=${#CELL_LINES[@]}
if (( ! N_TASKS )); then
    echo "no cells for STAGE=${STAGE}." >&2
    echo "  scripts/lr_grid_cells.py produced nothing. Run it directly to see" >&2
    echo "  why; the usual causes are the environment not being active," >&2
    echo "  FIT=${FIT} missing, or for stage 3 the LR law not fitted yet." >&2
    exit 1
fi

# One line per cell: d_model, slot layers, examples per epoch, beta2, epochs,
# seed, and a fixed learning rate or "-" for "search the grid around the prior".
cell_of() {   # $1 = task index
    read -r D_MODEL SLOT_ENCODER_LAYERS LIMIT_EXAMPLES BETA2 EPOCHS SEED CELL_LR \
        <<< "${CELL_LINES[$1]}"
}

# Everything past here trains, so torch has to be importable. Fail now rather
# than after the scheduler has handed out a GPU and lr_tuning.py has loaded a
# dataset.
python3 -c 'import torch' 2>/dev/null || {
    echo "[ERROR] torch is not importable after loading the environment." >&2
    echo "        module: python/pytorch_251_311_cu124" >&2
    echo "        venv:   $HOME/envs/llm-env" >&2
    echo "        Failing before this job spends any more of the allocation." >&2
    exit 1
}

TASK="${SLURM_ARRAY_TASK_ID:?submit as an array job; run with --list to see the cells}"
if (( TASK >= N_TASKS )); then
    echo "task ${TASK} is past the ${N_TASKS} cells in this grid" >&2
    exit 1
fi
cell_of "${TASK}"
# Head count from the sweep's own policy, so a cell is the same model the
# sweep trains. Integer division by 32 agreed with it on every multiple of 32
# but gave d40 one 40-wide head where the sweep gives it five of 8.
N_HEADS="$(python3 -c 'import sys; sys.path.insert(0, ".")
from src.scaling.configs import n_heads_for; print(n_heads_for(int(sys.argv[1])))' \
    "${D_MODEL}")"

# Centre the LR window on the prior for THIS model size.
read -r N_PARAMS LR_PRIOR LR_LO LR_HI <<< "$(python3 - "${D_MODEL}" \
    "${SLOT_ENCODER_LAYERS}" "${N_HEADS}" "${LR_SPAN_DOWN}" "${LR_SPAN_UP}" <<'PYEOF'
import sys
sys.path.insert(0, ".")
from src.scaling.flops import ArchSpec, n_params
from src.scaling.configs import lr_for

d_model, se, n_heads = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
down, up = float(sys.argv[4]), float(sys.argv[5])
cfg = ArchSpec(d_model=d_model, n_heads=n_heads, head_mode="cross_attn",
               slot_encoder_layers=se, decoder_layers=1)
n = n_params(cfg)
prior = lr_for(n)
print(f"{n} {prior:.6e} {prior / down:.6e} {prior * up:.6e}")
PYEOF
)"
LR_MIN="${LR_MIN:-${LR_LO}}"
LR_MAX="${LR_MAX:-${LR_HI}}"
# A cell that carries its own rate trains at exactly that rate, once.
if [[ "${CELL_LR}" != "-" ]]; then
    LR_MIN="${CELL_LR}"; LR_MAX="${CELL_LR}"; N_LRS=1
elif (( N_LRS == 1 )); then
    # One point of logspace(lo, hi, 1) is lo, prior/LR_SPAN_DOWN, not the prior.
    # The first shard-aligned probe trained at 2.0e-5 that way and learned
    # nothing (DeltaE 41). Harmless for a throughput number, but a run should
    # train at the rate it says it does.
    LR_MIN="${LR_PRIOR}"; LR_MAX="${LR_PRIOR}"
fi

echo "=================================================================="
echo " STAGE ${STAGE}, cell ${TASK} of ${N_TASKS}"
echo "   d_model / se        ${D_MODEL} / ${SLOT_ENCODER_LAYERS}  (n_heads ${N_HEADS})"
echo "   N (parameters)      ${N_PARAMS}"
echo "   prior lr            ${LR_PRIOR}  (centre of the grid)"
echo "   D (examples)        ${LIMIT_EXAMPLES} x ${EPOCHS} epoch(s)"
echo "   seed                ${SEED}"
echo "   batch size          ${BATCH_SIZE}  (fixed; from configs.DEFAULT_BATCH_SIZE)"
echo "   AdamW beta2         ${BETA2}"
echo "   LR grid             ${N_LRS} points, ${LR_MIN} to ${LR_MAX}"
echo "   selection metric    ${SELECTION_METRIC}"
echo "   LR schedule         ${LR_SCHEDULE}"
[[ -n "${EVAL_FRACTIONS}" ]] && echo "   mid-run evals at    ${EVAL_FRACTIONS} of the run"
echo "=================================================================="

mkdir -p "${OUTPUT_DIR}"

# --skip-existing unless FORCE=1: a resubmitted array must never clobber a
# finished cell.
OVERWRITE_FLAG="--skip-existing"
[[ "${FORCE:-0}" == "1" ]] && OVERWRITE_FLAG="--force"

srun python scripts/lr_tuning.py \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" --seed "${SEED}" \
    --limit-examples "${LIMIT_EXAMPLES}" --limit-shard-aligned \
    --limit-val-examples "${LIMIT_VAL_EXAMPLES:-10000}" \
    --limit-de-examples "${LIMIT_DE_EXAMPLES:-2048}" \
    --selection-metric "${SELECTION_METRIC}" \
    --lr-min "${LR_MIN}" --lr-max "${LR_MAX}" --n-lrs "${N_LRS}" \
    --beta1 "${BETA1:-0.9}" --beta2 "${BETA2}" \
    --lr-schedule "${LR_SCHEDULE}" \
    ${EVAL_FRACTIONS:+--eval-fractions ${EVAL_FRACTIONS}} \
    --feature-mode raw_spectrum \
    --encoder-hidden 128 --encoder-out 64 --encoder-dropout 0.1 \
    --d-model "${D_MODEL}" --n-layers "${N_LAYERS:-8}" --dropout 0.1 \
    --head-mode "${HEAD_MODE}" --n-heads "${N_HEADS}" \
    --slot-encoder-layers "${SLOT_ENCODER_LAYERS}" --decoder-layers 1 \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS:-6}" --prefetch-factor 1 \
    --weight-decay 0.01 --grad-clip 1.0 --warmup-fraction 0.02 \
    --output-dir "${OUTPUT_DIR}" \
    "${OVERWRITE_FLAG}" \
    --log-every 100 --streaming --bf16 --plot \
    $([[ "${PER_EXAMPLE_DE:-1}" == "1" ]] && echo --per-example-de)

echo "[INFO] STAGE ${STAGE} cell ${TASK} done."
if [[ "${STAGE}" == "3" ]]; then
    echo "  Once stage 3 has landed, assemble and fit the final IsoFLOP:"
    echo "  python scripts/collect_isoflop.py --stage2-dir ${STAGE2_DIR} \\"
    echo "      --stage3-dir ${OUTPUT_DIR} --beta2 ${BETA2_WINNER} --rungs ${RUNGS}"
else
    echo "  Fit once the stage has landed:"
    echo "  python scripts/fit_lr_law.py --results-dir ${OUTPUT_DIR} \\"
    echo "      --coverage-from analyses/scaling/results/isoflop_fit.json"
fi
