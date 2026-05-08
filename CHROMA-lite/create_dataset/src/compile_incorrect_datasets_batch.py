#!/usr/bin/env python3
"""
compile_datasets_batch.py

Pure OpenAI Batch workflow wired to your existing helpers:
- Resolves input with get_input_dir(num_layers, incidence_angle, structure_seed)
- Saves sidecar & final parquet next to get_output_dir(...)
- One seed (structure_seed) => one batch
- Deterministic via per-seed RNG

USAGE
-----
# Submit one seed as a batch
python compile_datasets_batch.py submit \
  --num_layers 4 \
  --incidence_angle 30 \
  --structure_seed 42 \
  --model gpt-5-mini-2025-10-01 \
  --temperature 1.0

# Check status later
python compile_datasets_batch.py status --batch-id batch_abc123

# Collect (writes to get_output_dir(...))
python compile_datasets_batch.py collect \
  --num_layers 4 \
  --incidence_angle 30 \
  --structure_seed 42 \
  --batch-id batch_abc123
"""

from __future__ import annotations
import os, io, json, argparse, random
from typing import List, Dict, Any, Iterable, Tuple
from dataclasses import dataclass

import pandas as pd
from openai import OpenAI

# --- Your original logic & paths (assumed available) ---
from compile_datasets import (
    load_structures,
    transform_to_sRGB,
    parse_materials,
    parse_thicknesses
)
import incorrect_procedural_template

# Import only what we need from Rephraser for batch construction
from rephraser import Rephraser, DEVELOPER_PROMPTS
choose_prompt_name = Rephraser._choose_prompt_name

def get_input_dir(num_layers, incidence_angle, seed):
    input_dir = 'data_struct'
    input_dir = os.path.join(input_dir, f'layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi')
    input_dir = os.path.join(input_dir, f"TR_simulations_layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi_seed_{seed:05d}.parquet")

    return input_dir

def get_output_dir(num_layers, incidence_angle, seed):
    output_dir = 'data_prompts'
    output_dir = os.path.join(output_dir, f'incorrect_prompts')
    output_dir = os.path.join(output_dir, f"TR_simulations_layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi_seed_{seed:05d}.parquet")
    return output_dir

# ---------- Data container ----------

@dataclass(frozen=True)
class StructureItem:
    idx: int
    structure_seed: int
    materials: List[str]
    thicknesses: List[int]
    sRGB_R: List[int]          # 3 ints (0..255)
    incidence_angle: int
    num_layers: int
    user_prompt: str
    prompt_seed: int  # The single seed used for both template and dev prompt selection


# ---------- Helpers ----------

def get_client() -> OpenAI:
    api_key = os.environ["CHROMA_KEY"]
    return OpenAI(api_key=api_key)

def iter_items_from_seed(
    input_parquet_path: str, *,
    num_layers: int,
    incidence_angle: int,
    structure_seed: int
) -> Iterable[StructureItem]:
    """
    Uses your original loader to produce structures.
    Generates ONE prompt_seed per structure, deterministically from structure_seed.
    This seed is used for BOTH the procedural template AND developer prompt selection.
    """
    structures = load_structures(input_parquet_path)

    # Deterministic per-seed RNG
    prompt_seed_rng = random.Random(structure_seed * 1_000_000 + num_layers * 1_000 + incidence_angle)

    for idx, (materials_str, thicknesses_str, numeric) in enumerate(structures):
        materials = parse_materials(materials_str)
        thicknesses = parse_thicknesses(thicknesses_str)

        wavelengths_m = numeric[:, 0]
        R = numeric[:, 1]

        # Convert to nm and compute sRGB via your function
        wavelengths_nm = (wavelengths_m * 1e9)
        sRGB_R = transform_to_sRGB(wavelengths_nm, R)

        # Generate ONE seed for this structure
        prompt_seed = prompt_seed_rng.randint(0, 2**32 - 1)
        
        # Build your exact prompt using this seed
        user_prompt = incorrect_procedural_template.get_template_with_errors(
            num_layers, materials, thicknesses, incidence_angle, sRGB_R, prompt_seed, verbose=False
        )

        yield StructureItem(
            idx=idx,
            structure_seed=structure_seed,
            materials=materials,
            thicknesses=thicknesses,
            sRGB_R=sRGB_R,
            incidence_angle=incidence_angle,
            num_layers=num_layers,
            user_prompt=user_prompt,
            prompt_seed=prompt_seed,
        )

def build_jsonl_and_sidecar(
    items: Iterable[StructureItem],
    model: str,
    temperature: float,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Returns (jsonl_lines, sidecar_df).
    custom_id = f"{structure_seed * 1_000_000 + num_layers * 1_000 + incidence_angle}:{idx:06d}"
    
    Each batch request mimics what Rephraser.rephrase() does:
    - Uses the same prompt_seed for both developer prompt selection and API seed
    - Constructs messages with developer prompt + user prompt (template)
    """
    lines: List[Dict[str, Any]] = []
    sidecar_rows: List[Dict[str, Any]] = []

    for it in items:
        # Use the prompt_seed from the item (already generated deterministically)
        dev_name = choose_prompt_name(it.prompt_seed)
        dev_prompt = DEVELOPER_PROMPTS[dev_name]
        custom_id = f"{it.structure_seed * 1_000_000 + it.num_layers * 1_000 + it.incidence_angle}:{it.idx:06d}"
        #This custom id is uniue for every structure, first part is used to generate prompt seed

        lines.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "seed": it.prompt_seed,  # Same seed for API reproducibility
                "messages": [
                    {"role": "developer", "content": [{"type": "text", "text": dev_prompt}]},
                    {"role": "developer", "content": [{"type": "text", "text": "Note: The description may contain contradictions or physically impossible constraints. These are intentional. Do not fix, remove, or reinterpret them — include them unchanged in your rephrasing.\n\n"}]},
                    {"role": "user", "content": [{"type": "text", "text": it.user_prompt}]},
                ],
            },
        })

        sidecar_rows.append({
            "custom_id": custom_id,
            "structure_seed": it.structure_seed,
            "idx": it.idx,
            "prompt_seed": it.prompt_seed,
            "dev_prompt_name": dev_name,
            "num_layers": it.num_layers,
            "incidence_angle": it.incidence_angle,
            "materials": json.dumps(it.materials, ensure_ascii=False),
            "thicknesses": json.dumps(it.thicknesses, ensure_ascii=False),
            "sRGB_R": json.dumps(it.sRGB_R, ensure_ascii=False),
        })

    return lines, pd.DataFrame(sidecar_rows)

def upload_and_create_batch(client: OpenAI, lines: List[Dict[str, Any]], metadata: Dict[str, str]) -> str:
    """
    Upload JSONL (purpose='batch') and create the batch. Returns batch_id.
    """
    buf = io.BytesIO()
    for line in lines:
        buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
    buf.seek(0)

    f = client.files.create(file=("seed.jsonl", buf, "application/jsonl"), purpose="batch")
    b = client.batches.create(
        input_file_id=f.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata,
    )
    return b.id

def download_batch_output(client: OpenAI, batch_id: str) -> pd.DataFrame:
    """
    Batch must already be completed. Returns DataFrame with:
    custom_id, text | error, model, system_fingerprint, finish_reason
    """
    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        raise RuntimeError(f"Batch {batch_id} not completed (status={b.status})")

    content = client.files.content(b.output_file_id).read().decode("utf-8")

    rows = []
    for line in content.splitlines():
        obj = json.loads(line)
        if "response" in obj:
            body = obj["response"]["body"]
            ch0 = body["choices"][0]
            rows.append({
                "custom_id": obj["custom_id"],
                "text": ch0["message"]["content"],
                "finish_reason": ch0.get("finish_reason"),
                "model": body.get("model"),
                "system_fingerprint": body.get("system_fingerprint"),
            })
        else:
            rows.append({
                "custom_id": obj.get("custom_id"),
                "error": obj.get("error"),
            })
    return pd.DataFrame(rows)


# ---------- Commands ----------

def cmd_submit(args):
    client = get_client()

    # Resolve paths using your helpers
    input_parquet_path = get_input_dir(args.num_layers, args.incidence_angle, args.structure_seed)
    output_parquet_path = get_output_dir(args.num_layers, args.incidence_angle, args.structure_seed)
    out_dir = os.path.dirname(output_parquet_path)
    os.makedirs(out_dir, exist_ok=True)

    # Build items using your original loader + transform + template
    items = list(iter_items_from_seed(
        input_parquet_path,
        num_layers=args.num_layers,
        incidence_angle=args.incidence_angle,
        structure_seed=args.structure_seed,
    ))

    lines, sidecar_df = build_jsonl_and_sidecar(
        items, args.model, args.temperature
    )

    # Sidecar lives next to the final parquet
    sidecar_path = os.path.join(
        out_dir, f"seed_{args.structure_seed}_L{args.num_layers}_A{args.incidence_angle}.sidecar.csv"
    )
    sidecar_df.to_csv(sidecar_path, index=False)

    batch_id = upload_and_create_batch(
        client,
        lines,
        metadata={
            "num_layers": str(args.num_layers),
            "incidence_angle": str(args.incidence_angle),
            "structure_seed": str(args.structure_seed),
            "input": os.path.basename(input_parquet_path),
            "output": os.path.basename(output_parquet_path),
        },
    )

    print(f"[SUBMITTED] seed={args.structure_seed}")
    print(f"  batch_id: {batch_id}")
    print(f"  sidecar:  {sidecar_path}")
    print(f"  output:   {output_parquet_path}")

def cmd_status(args):
    client = get_client()
    b = client.batches.retrieve(args.batch_id)

    # If available, prefer model_dump_json (most robust)
    try:
        print(b.model_dump_json(indent=2))
        return
    except AttributeError:
        pass

    # Otherwise, build a plain dict and serialize
    out = {
        "id": b.id,
        "status": b.status,
        "created_at": b.created_at,
        "in_progress_at": getattr(b, "in_progress_at", None),
        "completed_at": getattr(b, "completed_at", None),
        "request_counts": (
            b.request_counts.model_dump()
            if hasattr(b.request_counts, "model_dump") else
            dict(b.request_counts) if hasattr(b.request_counts, "__iter__") else
            str(b.request_counts)
        ),
        "metadata": getattr(b, "metadata", None),
    }
    print(json.dumps(out, indent=2, default=str))


def cmd_collect(args):
    client = get_client()

    # Resolve where to write and where the sidecar is, using your helpers
    output_parquet_path = get_output_dir(args.num_layers, args.incidence_angle, args.structure_seed)
    out_dir = os.path.dirname(output_parquet_path)
    sidecar_path = os.path.join(
        out_dir, f"seed_{args.structure_seed}_L{args.num_layers}_A{args.incidence_angle}.sidecar.csv"
    )

    out_df = download_batch_output(client, args.batch_id)
    sidecar = pd.read_csv(sidecar_path)

    merged = sidecar.merge(out_df, on="custom_id", how="left")

    os.makedirs(out_dir, exist_ok=True)
    # NOTE: this writes to your canonical output path
    merged.to_parquet(output_parquet_path, index=False)

    print(f"[COLLECTED] wrote {output_parquet_path}")
    print(f"  sidecar:   {sidecar_path}")

# ---------- CLI ----------

def build_parser():
    p = argparse.ArgumentParser(description="Pure batch compiler (one seed => one batch).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # submit
    ps = sub.add_parser("submit", help="Submit a batch for one seed.")
    ps.add_argument('--num_layers', type=int, required=True)
    ps.add_argument('--incidence_angle', type=int, required=True)
    ps.add_argument('--structure_seed', type=int, required=True)
    ps.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    ps.add_argument("--temperature", type=float, default=1.0)
    ps.set_defaults(func=cmd_submit)

    # status
    pst = sub.add_parser("status", help="Check batch status.")
    pst.add_argument("--batch-id", type=str, required=True)
    pst.set_defaults(func=cmd_status)

    # collect
    pc = sub.add_parser("collect", help="Collect a completed batch into Parquet.")
    pc.add_argument('--num_layers', type=int, required=True)
    pc.add_argument('--incidence_angle', type=int, required=True)
    pc.add_argument('--structure_seed', type=int, required=True)
    pc.add_argument("--batch-id", type=str, required=True)
    pc.set_defaults(func=cmd_collect)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()