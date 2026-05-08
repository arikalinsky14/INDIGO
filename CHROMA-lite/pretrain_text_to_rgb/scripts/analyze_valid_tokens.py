#!/usr/bin/env python3
"""
Analyze TinyLlama/Llama-2 tokenizer to identify the MINIMAL set of valid
tokens for constrained [R,G,B] output, and build the compact vocabulary
mapping needed for lm_head weight surgery.

Output format: [DDD,DDD,DDD]
Valid token classes (strictly single-character when stripped):
  - Single digits: 0-9
  - Brackets: [ ]
  - Comma: ,
  - Whitespace: space tokens (allowed but stripped in post-processing)
  - EOS: end of sequence

We EXCLUDE:
  - Compound tokens like '[]', ']]', '][', ',,', '],', etc.
  - Multi-digit number tokens like '12', '255'
  - Token 11167 which decodes to ',\\r' (comma + carriage return)

Usage:
    python analyze_valid_tokens.py
    python analyze_valid_tokens.py --save valid_token_vocab.pt
    python analyze_valid_tokens.py --encoder meta-llama/Llama-2-7b-chat-hf
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer

# Explicitly blocked token IDs (discovered during analysis)
BLOCKED_TOKEN_IDS = {
    11167,  # ',\r' — comma with carriage return
}


def analyze_tokenizer(encoder_name: str) -> dict:
    """
    Identify the minimal valid token set for [R,G,B] constrained decoding.

    Returns:
        dict with valid_ids, compact_vocab mapping, categories, etc.
    """
    print(f"[INFO] Loading tokenizer: {encoder_name}")
    tok = AutoTokenizer.from_pretrained(encoder_name)

    vocab = tok.get_vocab()
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    print(f"[INFO] Vocabulary size: {vocab_size}")
    print(f"[INFO] EOS: {repr(tok.eos_token)} (id={tok.eos_token_id})")

    VALID_SINGLE_CHARS = set("0123456789[],")

    categories = defaultdict(list)
    valid_ids = set()
    multi_digit_numbers = []

    for token_id in range(vocab_size):
        decoded = tok.decode([token_id], skip_special_tokens=False)
        raw = inv_vocab.get(token_id, "")
        stripped = decoded.strip()

        # --- Explicitly blocked ---
        if token_id in BLOCKED_TOKEN_IDS:
            categories['blocked'].append((token_id, decoded, raw))
            continue

        # --- Special tokens ---
        if token_id == tok.eos_token_id:
            categories['eos'].append((token_id, decoded, raw))
            valid_ids.add(token_id)
            continue
        if token_id in [tok.bos_token_id, 0]:
            categories['excluded_special'].append((token_id, decoded, raw))
            continue

        # --- Pure whitespace ---
        if decoded and all(c == ' ' for c in decoded):
            categories['whitespace'].append((token_id, decoded, raw))
            valid_ids.add(token_id)
            continue

        if not stripped:
            continue

        # Single digit?
        if len(stripped) == 1 and stripped in "0123456789":
            categories['digit'].append((token_id, decoded, raw))
            valid_ids.add(token_id)
            continue

        # Single bracket?
        if stripped == '[':
            categories['bracket_open'].append((token_id, decoded, raw))
            valid_ids.add(token_id)
            continue
        if stripped == ']':
            categories['bracket_close'].append((token_id, decoded, raw))
            valid_ids.add(token_id)
            continue

        # Single comma?
        if stripped == ',':
            categories['comma'].append((token_id, decoded, raw))
            valid_ids.add(token_id)
            continue

        # --- Track but EXCLUDE multi-digit numbers ---
        if stripped.isdigit() and len(stripped) > 1:
            multi_digit_numbers.append((token_id, stripped, raw))
            continue

        # --- Track but EXCLUDE compound tokens ---
        if all(c in VALID_SINGLE_CHARS for c in stripped) and len(stripped) > 1:
            categories['excluded_compound'].append((token_id, decoded, raw))
            continue

    # ================================================================
    # REPORT
    # ================================================================

    print(f"\n{'='*70}")
    print(f"STRICT VALID TOKEN SET")
    print(f"{'='*70}")
    print(f"Total vocabulary:  {vocab_size}")
    print(f"Valid tokens:      {len(valid_ids)}")
    print(f"Reduction:         {vocab_size} -> {len(valid_ids)} "
          f"({len(valid_ids)/vocab_size*100:.2f}%)")

    for cat in ['digit', 'bracket_open', 'bracket_close', 'comma', 'whitespace', 'eos']:
        items = categories.get(cat, [])
        print(f"\n  {cat} ({len(items)} tokens):")
        for tid, decoded, raw in sorted(items, key=lambda x: x[0]):
            print(f"    {tid:>6}  decoded={repr(decoded):<20}  raw={repr(raw)}")

    blocked = categories.get('blocked', [])
    if blocked:
        print(f"\n  blocked ({len(blocked)} tokens):")
        for tid, decoded, raw in sorted(blocked, key=lambda x: x[0]):
            print(f"    {tid:>6}  decoded={repr(decoded):<20}  raw={repr(raw)}")

    print(f"\n{'='*70}")
    print(f"MULTI-DIGIT NUMBER TOKENS (EXCLUDED)")
    print(f"{'='*70}")
    if multi_digit_numbers:
        print(f"  Found {len(multi_digit_numbers)} multi-digit number tokens:")
        for tid, val, raw in sorted(multi_digit_numbers, key=lambda x: (len(x[1]), int(x[1]))):
            in_range = "<=255" if int(val) <= 255 else ">255"
            print(f"    {tid:>6}  value={val:<10}  raw={repr(raw):<20}  {in_range}")
    else:
        print("  None found - tokenizer uses single-digit tokenization")

    excluded = categories.get('excluded_compound', [])
    if excluded:
        print(f"\n  Excluded compound tokens ({len(excluded)}):")
        for tid, decoded, raw in sorted(excluded, key=lambda x: x[0]):
            print(f"    {tid:>6}  decoded={repr(decoded):<20}  raw={repr(raw)}")

    # ================================================================
    # VERIFY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"TOKENIZATION VERIFICATION")
    print(f"{'='*70}")

    examples = [
        "[113,234,34]",
        "[0,0,0]",
        "[255,255,255]",
        "[128,64,192]",
        "[1,2,3]",
    ]

    all_ok = True
    for ex in examples:
        ids = tok.encode(ex, add_special_tokens=False)
        tokens = [repr(tok.decode([i])) for i in ids]
        missing = [i for i in ids if i not in valid_ids]
        status = "OK" if not missing else "FAIL"
        if missing:
            all_ok = False

        print(f"  {status} {ex:<25} -> {ids}")
        print(f"    tokens: {tokens}")
        if missing:
            for m in missing:
                print(f"    MISSING: {m} = {repr(tok.decode([m]))} raw={repr(inv_vocab.get(m))}")

    if all_ok:
        print(f"\n  All examples tokenize cleanly with the valid set")
    else:
        print(f"\n  WARNING: Some tokens missing!")

    # ================================================================
    # COMPACT VOCABULARY
    # ================================================================

    sorted_valid = sorted(valid_ids)
    orig_to_compact = {orig: compact for compact, orig in enumerate(sorted_valid)}
    compact_to_orig = {compact: orig for compact, orig in enumerate(sorted_valid)}

    print(f"\n{'='*70}")
    print(f"COMPACT VOCABULARY FOR lm_head SURGERY")
    print(f"{'='*70}")
    print(f"  Compact vocab size: {len(sorted_valid)}")
    print(f"  lm_head reduction:  Linear(hidden, {vocab_size}) -> Linear(hidden, {len(sorted_valid)})")
    hidden_dim = 2048
    print(f"  Parameter reduction: {vocab_size * hidden_dim:,} -> {len(sorted_valid) * hidden_dim:,} "
          f"({len(sorted_valid) * hidden_dim / (vocab_size * hidden_dim) * 100:.3f}%)")

    print(f"\n  Compact ID mapping:")
    for compact_id, orig_id in compact_to_orig.items():
        decoded = tok.decode([orig_id], skip_special_tokens=False).strip() or repr(tok.decode([orig_id]))
        print(f"    compact {compact_id:>3} -> orig {orig_id:>6}  ({decoded})")

    print(f"\n  Round-trip test: '[128,64,192]'")
    orig_ids = tok.encode("[128,64,192]", add_special_tokens=False)
    compact_ids = [orig_to_compact[oid] for oid in orig_ids]
    recovered = [compact_to_orig[cid] for cid in compact_ids]
    print(f"    original IDs: {orig_ids}")
    print(f"    compact IDs:  {compact_ids}")
    print(f"    recovered:    {recovered}")
    print(f"    match: {'yes' if recovered == orig_ids else 'NO'}")

    return {
        'valid_ids': sorted_valid,
        'orig_to_compact': orig_to_compact,
        'compact_to_orig': compact_to_orig,
        'categories': dict(categories),
        'multi_digit_numbers': multi_digit_numbers,
        'vocab_size': vocab_size,
        'compact_vocab_size': len(sorted_valid),
        'encoder_name': encoder_name,
        'eos_compact_id': orig_to_compact[tok.eos_token_id],
        'eos_orig_id': tok.eos_token_id,
    }


def save_vocab(result: dict, output_path: str):
    """Save compact vocabulary for lm_head surgery."""
    valid_ids = torch.tensor(result['valid_ids'], dtype=torch.long)
    n_valid = len(result['valid_ids'])
    vocab_size = result['vocab_size']

    orig_to_compact = torch.full((vocab_size,), -1, dtype=torch.long)
    for orig, compact in result['orig_to_compact'].items():
        orig_to_compact[orig] = compact

    compact_to_orig = torch.tensor(
        [result['compact_to_orig'][i] for i in range(n_valid)],
        dtype=torch.long
    )

    save_dict = {
        'valid_token_ids': valid_ids,
        'orig_to_compact': orig_to_compact,
        'compact_to_orig': compact_to_orig,
        'lm_head_row_indices': valid_ids,
        'n_valid': n_valid,
        'vocab_size': vocab_size,
        'encoder_name': result['encoder_name'],
        'eos_compact_id': result['eos_compact_id'],
        'eos_orig_id': result['eos_orig_id'],
    }

    torch.save(save_dict, output_path)
    print(f"\n[INFO] Saved compact vocab to {output_path}")
    print(f"       {n_valid} valid tokens / {vocab_size} total")


def main():
    parser = argparse.ArgumentParser(description='Build minimal valid token vocab for [R,G,B] decoding')
    parser.add_argument('--encoder', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--save-json', type=str, default=None)
    args = parser.parse_args()

    result = analyze_tokenizer(args.encoder)

    if args.save:
        save_vocab(result, args.save)

    if args.save_json:
        json_result = {
            'encoder_name': result['encoder_name'],
            'vocab_size': result['vocab_size'],
            'compact_vocab_size': result['compact_vocab_size'],
            'valid_ids': result['valid_ids'],
            'orig_to_compact': {str(k): v for k, v in result['orig_to_compact'].items()},
            'compact_to_orig': {str(k): v for k, v in result['compact_to_orig'].items()},
            'eos_compact_id': result['eos_compact_id'],
            'eos_orig_id': result['eos_orig_id'],
        }
        with open(args.save_json, 'w') as f:
            json.dump(json_result, f, indent=2)
        print(f"[INFO] JSON saved to {args.save_json}")


if __name__ == '__main__':
    main()