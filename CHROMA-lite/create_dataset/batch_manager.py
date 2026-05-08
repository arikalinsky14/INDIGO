#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
batch_manager.py
- Submit a grid of Batch API jobs via compile_datasets_batch.py
- Persist dynamic batch_ids per (num_layers, angle, seed)
- Poll status and collect completed jobs
- Skip combos whose final parquet already exists

Usage examples:
  python batch_manager.py submit-grid
  python batch_manager.py status-all
  python batch_manager.py collect-ready
  python batch_manager.py collect-all

Edit the PARAMS section below to choose which L/A/S to run.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from glob import glob

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "src" / "compile_datasets_batch.py"  # path to your batch script
STATE = ROOT / "batches_state.json"                  # stores batch_ids & metadata

# Default model for Batch (override via CLI in submit-grid if you want)
DEFAULT_MODEL = "gpt-4o-mini-2024-07-18"

# ---------- PARAMS: EDIT THESE ----------
LAYERS = [8]            # e.g., [4, 6, 8]
ANGLES = [0]            # e.g., [0, 30, 60]
SEEDS  = list(range(42, 4201, 42))  # add more as you like

# If your folder naming differs, adjust this glob to match how your script writes outputs
# Example observed:
# data_prompts\layers_04_angle_00_substrate_CSi\TR_simulations_layers_04_angle_00_substrate_CSi_seed_00042.parquet
def output_glob(num_layers: int, angle: int, seed: int) -> str:
    return str(
        ROOT / "data_prompts" /
        f"layers_{num_layers:02d}_angle_{angle:02d}_substrate_*" /
        f"TR_simulations_layers_{num_layers:02d}_angle_{angle:02d}_substrate_*_seed_{seed:05d}.parquet"
    )

# ---------------------------------------

def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"jobs": {}}  # {(L,A,S): {"batch_id": "...", "status": "...", "model": "..."}}

def save_state(state):
    STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def key_tuple(num_layers, angle, seed):
    return f"{num_layers}-{angle}-{seed}"

def already_collected(num_layers, angle, seed) -> bool:
    files = glob(output_glob(num_layers, angle, seed))
    return len(files) > 0

def run_cmd(args, capture_json=False):
    """Run a python command. If capture_json=True, parse stdout as JSON."""
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    out = proc.stdout
    if capture_json:
        try:
            return json.loads(out)
        except Exception:
            # Try to find a JSON block in the output (in case extra prints surround it)
            m = re.search(r"(\{.*\})", out, flags=re.DOTALL)
            if m:
                return json.loads(m.group(1))
            raise
    return out

def submit_one(num_layers, angle, seed, model=DEFAULT_MODEL):
    """Submit a single batch; return batch_id or None."""
    # Skip if final parquet already exists
    if already_collected(num_layers, angle, seed):
        print(f"[SKIP-EXISTS] L={num_layers} A={angle} S={seed} -> final parquet already present")
        return None

    cmd = [
        sys.executable, str(SCRIPT), "submit",
        "--num_layers", str(num_layers),
        "--incidence_angle", str(angle),
        "--structure_seed", str(seed),
        "--model", model
    ]
    out = run_cmd(cmd, capture_json=False)
    # The submit script prints lines like:
    # [SUBMITTED] seed=42
    #   batch_id: batch_...
    # Extract the batch_id by regex:
    m = re.search(r"batch_id:\s*([a-zA-Z0-9_]+)", out)
    if not m:
        print(f"[ERROR] Could not parse batch_id from submit output for L={num_layers} A={angle} S={seed}\n{out}")
        return None
    batch_id = m.group(1)
    print(f"[SUBMITTED] L={num_layers} A={angle} S={seed} -> {batch_id}")
    return batch_id

def status_one(batch_id):
    cmd = [sys.executable, str(SCRIPT), "status", "--batch-id", batch_id]
    return run_cmd(cmd, capture_json=True)

def collect_one(num_layers, angle, seed, batch_id):
    # Skip if final parquet already exists
    if already_collected(num_layers, angle, seed):
        print(f"[SKIP-COLLECT] L={num_layers} A={angle} S={seed} -> final parquet already present")
        return True

    cmd = [
        sys.executable, str(SCRIPT), "collect",
        "--num_layers", str(num_layers),
        "--incidence_angle", str(angle),
        "--structure_seed", str(seed),
        "--batch-id", batch_id
    ]
    out = run_cmd(cmd, capture_json=False)
    print(f"[COLLECT] L={num_layers} A={angle} S={seed} ->\n{out}")
    # After collect, we expect the parquet to exist
    ok = already_collected(num_layers, angle, seed)
    return ok

# ---------------- COMMANDS ----------------

def cmd_submit_grid(model=DEFAULT_MODEL):
    state = load_state()
    for L in LAYERS:
        for A in ANGLES:
            for S in SEEDS:
                k = key_tuple(L, A, S)
                if k in state["jobs"] and state["jobs"][k].get("batch_id"):
                    print(f"[SKIP-HAVE-ID] L={L} A={A} S={S} -> {state['jobs'][k]['batch_id']}")
                    continue
                bid = submit_one(L, A, S, model=model)
                if bid:
                    state["jobs"][k] = {"batch_id": bid, "status": "submitted", "model": model}
                    save_state(state)
    print("[DONE] submit-grid")

def cmd_status_all():
    state = load_state()
    if not state["jobs"]:
        print("[INFO] No jobs in state file.")
        return
    for k, rec in state["jobs"].items():
        batch_id = rec.get("batch_id")
        if not batch_id:
            print(f"[WARN] {k} has no batch_id")
            continue
        try:
            st = status_one(batch_id)
        except Exception as e:
            print(f"[ERROR] status for {k} ({batch_id}): {e}")
            continue
        # Print summary and update state
        status = st.get("status")
        counts = st.get("request_counts", {})
        comp = counts.get("completed")
        tot  = counts.get("total")
        failed = counts.get("failed")
        print(f"[STATUS] {k} -> {status} | completed={comp}/{tot} failed={failed} id={batch_id}")
        rec["status"] = status
        rec["request_counts"] = counts
        rec["output_file_id"] = st.get("output_file_id")
        save_state(state)
    print("[DONE] status-all")

def cmd_collect_ready():
    state = load_state()
    if not state["jobs"]:
        print("[INFO] No jobs in state file.")
        return

    for k, rec in state["jobs"].items():
        batch_id = rec.get("batch_id")
        if not batch_id:
            continue

        # Refresh status
        try:
            st = status_one(batch_id)
        except Exception as e:
            print(f"[ERROR] status for {k} ({batch_id}): {e}")
            continue

        status = st.get("status")
        rec["status"] = status
        save_state(state)

        # Only collect if completed (and not yet present)
        parts = k.split("-")
        L, A, S = map(int, parts)
        if status == "completed":
            ok = collect_one(L, A, S, batch_id)
            if ok:
                rec["collected"] = True
                save_state(state)
        elif status == "failed":
            print(f"[FAILED] {k} ({batch_id}) — check errors via `status` output")
        else:
            print(f"[WAIT] {k} ({batch_id}) -> {status}")

    print("[DONE] collect-ready")

def cmd_collect_all():
    """Force collect for all known jobs (still skips if parquet already exists)."""
    state = load_state()
    if not state["jobs"]:
        print("[INFO] No jobs in state file.")
        return
    for k, rec in state["jobs"].items():
        batch_id = rec.get("batch_id")
        if not batch_id:
            continue
        L, A, S = map(int, k.split("-"))
        ok = collect_one(L, A, S, batch_id)
        if ok:
            rec["collected"] = True
            save_state(state)
    print("[DONE] collect-all")

def cmd_clear_failed():
    """Remove batch_id entries for any jobs marked as failed."""
    state = load_state()
    if not state["jobs"]:
        print("[INFO] No jobs in state file.")
        return

    removed = 0
    for k, rec in list(state["jobs"].items()):
        status = rec.get("status")
        if status == "failed":
            print(f"[CLEAR-FAILED] Removing failed batch for {k} ({rec.get('batch_id')})")
            rec.pop("batch_id", None)
            rec.pop("status", None)
            rec.pop("request_counts", None)
            rec.pop("output_file_id", None)
            rec.pop("collected", None)
            removed += 1

    if removed:
        save_state(state)
        print(f"[DONE] Cleared {removed} failed job(s).")
    else:
        print("[INFO] No failed jobs to clear.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_manager.py [submit|status|collect|force-collect-all|clear-failed] [--model NAME]")
        sys.exit(2)

    cmd = sys.argv[1]
    model = DEFAULT_MODEL

    if "--model" in sys.argv:
        i = sys.argv.index("--model")
        if i+1 < len(sys.argv):
            model = sys.argv[i+1]

    if cmd == "submit":
        cmd_submit_grid(model=model)
    elif cmd == "status":
        cmd_status_all()
    elif cmd == "collect":
        cmd_collect_ready()
    elif cmd == "force-collect-all":
        cmd_collect_all()
    elif cmd == "clear-failed":
        cmd_clear_failed()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(2)

if __name__ == "__main__":
    main()

