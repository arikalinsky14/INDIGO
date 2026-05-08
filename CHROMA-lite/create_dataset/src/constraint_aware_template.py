"""
Constraint-Aware Template Generator

Mirrors the logic of procedural_template.get_template() but also captures
structured constraint annotations alongside the natural language text.

This avoids modifying the original procedural_template.py. Instead, it
reimplements the template generation while recording each constraint in
the StructuredConstraints format defined in src/constraints.py.

Usage:
    text, constraints_dict = get_template_with_constraints(
        num_layers=4, materials=["SiO2", "Ag", "TiO2", "Au"],
        thicknesses=[100, 50, 75, 120], incidence_angle=0,
        color_rgb=[128, 64, 200], seed=42
    )
"""

from __future__ import annotations
import math
import copy
import random as _random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# Import from the existing procedural template module
import procedural_template as pt


def get_template_with_constraints(
    num_layers: int,
    materials: List[str],
    thicknesses: List[int],
    incidence_angle: int,
    color_rgb: List[int],
    seed: int = 42,
    # Higher constraint density for test set
    avg_adj_constraints: float = 0.2,
    avg_rel_tkns_constraints: float = 0.2,
    avg_indv_tkns_constraints: float = 1.5,
    ttl_tkns_constraint_prob: float = 0.8,
    lyr_id_constraint_prob: float = 0.15,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a natural language prompt and structured constraints.

    Returns:
        (text, constraints_dict) where constraints_dict matches the
        StructuredConstraints schema from src/constraints.py
    """
    # Seed the RNGs (same approach as procedural_template)
    rng = _random.Random(seed)
    np_rng = np.random.default_rng(seed)

    constraints: Dict[str, Any] = {
        "layer_count_type": "none",
        "layer_count_value": None,
        "materials_type": "any",
        "allowed_materials": [],
        "forbidden_materials": [],
        "additional": [],
    }

    # --- Layer count fragment ---
    text, layer_choice = _get_layers_fragment(num_layers, rng, np_rng)
    if layer_choice == "Exact":
        constraints["layer_count_type"] = "exact"
        constraints["layer_count_value"] = num_layers
    elif layer_choice == "Min":
        noise = int(np_rng.poisson(num_layers * 0.25))
        minimum = max(num_layers - noise, 2)
        constraints["layer_count_type"] = "min"
        constraints["layer_count_value"] = minimum
    elif layer_choice == "Max":
        noise = int(np_rng.poisson(num_layers * 0.25))
        maximum = min(num_layers + noise, 10)
        constraints["layer_count_type"] = "max"
        constraints["layer_count_value"] = maximum

    # --- Materials fragment ---
    mat_text, extra_materials, mat_type, allowed, forbidden = _get_materials_fragment(
        materials, rng, np_rng)
    text += mat_text
    constraints["materials_type"] = mat_type
    constraints["allowed_materials"] = allowed
    constraints["forbidden_materials"] = forbidden

    # --- Color identity ---
    text += _get_color_identity(color_rgb, rng)

    # --- Additional constraints (higher density for test set) ---
    additional_texts, additional_constraints = _get_additional_constraints(
        materials, thicknesses, extra_materials,
        rng, np_rng,
        avg_adj_constraints=avg_adj_constraints,
        avg_rel_tkns_constraints=avg_rel_tkns_constraints,
        avg_indv_tkns_constraints=avg_indv_tkns_constraints,
        ttl_tkns_constraint_prob=ttl_tkns_constraint_prob,
        lyr_id_constraint_prob=lyr_id_constraint_prob,
    )
    constraints["additional"] = additional_constraints

    # Shuffle and append
    combined = list(zip(additional_texts, range(len(additional_texts))))
    rng.shuffle(combined)
    shuffled_indices = [idx for _, idx in combined]
    for idx in shuffled_indices:
        text += additional_texts[idx]

    # Also reorder additional constraints to match shuffled text
    constraints["additional"] = [additional_constraints[idx] for idx in shuffled_indices]

    return text, constraints


# ============================================================================
# Internal helpers (mirror procedural_template.py logic)
# ============================================================================

def _get_layers_fragment(
    num_layers: int, rng: _random.Random, np_rng: np.random.Generator
) -> Tuple[str, str]:
    """Returns (text_fragment, choice_type)."""
    noise = int(np_rng.poisson(num_layers * 0.25))
    options = ["Min", "Max", "Exact", "None"]
    weights = [0.125, 0.125, 0.25, 0.50]
    choice = rng.choices(options, weights=weights, k=1)[0]

    if choice == "Min":
        minimum = max(num_layers - noise, 2)
        return (f"Create an optical structure that consists of a stack with at least "
                f"{pt.get_str_num_layers(minimum)} layers; ", choice)
    elif choice == "Max":
        maximum = min(num_layers + noise, 10)
        return (f"Create an optical structure that consists of a stack with at most "
                f"{pt.get_str_num_layers(maximum)} layers; ", choice)
    elif choice == "Exact":
        return (f"Create an optical structure that consists of "
                f"{pt.get_str_a_num_layers(num_layers)}-layer stack; ", choice)
    else:
        return "Create an optical structure that consists of a layered stack; ", choice


def _get_materials_fragment(
    materials: List[str], rng: _random.Random, np_rng: np.random.Generator
) -> Tuple[str, List[str], str, List[str], List[str]]:
    """Returns (text, extra_materials, type, allowed, forbidden)."""
    options = ["Strict", "Helpful", "Extra", "Restrict", "Any"]
    weights = [0.05, 0.05, 0.30, 0.30, 0.30]
    choice = rng.choices(options, weights=weights, k=1)[0]

    if choice == "Strict":
        shuffled = _shuffle_materials(materials, np_rng)
        text = f"the stack must be composed of {', '.join(shuffled)}. "
        return text, [], "strict", list(materials), []

    elif choice == "Helpful":
        shuffled = _shuffle_materials(materials, np_rng)
        text = (f"the stack may be composed of {', '.join(shuffled)}, "
                f"but usage of all materials is not required. ")
        return text, [], "helpful", list(materials), []

    elif choice == "Extra":
        extra_count = min(int(np_rng.poisson(len(pt.get_unused_materials(materials)) * 0.2)), 9) + 1
        extra_materials = rng.sample(pt.get_unused_materials(materials), extra_count)
        new_materials = materials + extra_materials
        shuffled = _shuffle_materials(new_materials, np_rng)
        text = (f"the stack may be composed of {', '.join(shuffled)}, "
                f"but usage of all materials is not required. ")
        return text, extra_materials, "extra", list(new_materials), []

    elif choice == "Restrict":
        unused = pt.get_unused_materials(materials)
        restrict_count = max(min(int(np_rng.poisson(len(unused) * 0.3)), len(unused)), 1)
        restricted = rng.sample(unused, restrict_count)
        remaining = [m for m in unused if m not in restricted]
        shuffled = _shuffle_materials(restricted, np_rng)
        text = f"the stack may be composed of any material except {', '.join(shuffled)}. "
        return text, remaining, "restrict", [], list(restricted)

    else:  # Any
        extra_materials = pt.get_unused_materials(materials)
        text = "the stack may be composed of any material. "
        return text, extra_materials, "any", [], []


def _shuffle_materials(materials: List[str], np_rng: np.random.Generator) -> List[str]:
    """Shuffle a copy of materials list."""
    copied = copy.deepcopy(materials)
    np_rng.shuffle(copied)
    return list(copied)


def _get_color_identity(color_rgb: List[int], rng: _random.Random) -> str:
    """Generate color description text."""
    options = ["RGB", "Name", "Specific Name", "HEX"]
    weights = [1/3, 1/6, 1/6, 1/3]
    choice = rng.choices(options, weights=weights, k=1)[0]
    r, g, b = map(int, color_rgb)

    if choice == "RGB":
        return (f"The reflected color observed from this multilayer configuration "
                f"must correspond to the RGB value ({r}, {g}, {b}). ")
    elif choice == "Name":
        from color_palette_manager import PaletteMatcher as PM
        common_name = PM.find(r, g, b)
        return (f"The reflected color observed from this multilayer configuration "
                f"must correspond to {common_name}. ")
    elif choice == "Specific Name":
        from color_palette_manager import PaletteMatcher as PM
        with PM.use("__full__"):
            full_name = PM.find(r, g, b)
        return (f"The reflected color observed from this multilayer configuration "
                f"must correspond to {full_name}. ")
    elif choice == "HEX":
        hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return (f"The reflected color observed from this multilayer configuration "
                f"must correspond to the HEX code {hex_code}. ")
    return ""


def _get_additional_constraints(
    materials: List[str],
    thicknesses: List[int],
    extra_materials: List[str],
    rng: _random.Random,
    np_rng: np.random.Generator,
    avg_adj_constraints: float = 0.2,
    avg_rel_tkns_constraints: float = 0.2,
    avg_indv_tkns_constraints: float = 1.5,
    ttl_tkns_constraint_prob: float = 0.8,
    lyr_id_constraint_prob: float = 0.15,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Generate additional constraints and return both text fragments and
    structured constraint dicts.

    Returns:
        (text_fragments, constraint_dicts) - parallel lists
    """
    all_mats = materials + extra_materials
    all_thks = thicknesses + [
        max(thicknesses[rng.randrange(len(thicknesses))] +
            (int(np_rng.poisson(thicknesses[rng.randrange(len(thicknesses))] * 0.25 / 5)) * 5 + 5)
            * rng.choice([-1, 1]), 5)
        for _ in extra_materials
    ]

    texts: List[str] = []
    constraints: List[Dict[str, Any]] = []

    # Adjacency constraints
    n_adj = int(np_rng.poisson(avg_adj_constraints))
    for _ in range(n_adj):
        if len(all_mats) < 2:
            break
        mat_a, mat_b = rng.sample(all_mats, 2)
        a_in = mat_a in materials
        b_in = mat_b in materials
        if a_in and b_in:
            idx_a = materials.index(mat_a)
            idx_b = materials.index(mat_b)
            is_adj = abs(idx_a - idx_b) == 1
        else:
            idx_a = all_mats.index(mat_a)
            idx_b = all_mats.index(mat_b)
            is_adj = abs(idx_a - idx_b) == 1

        if is_adj:
            texts.append(f"If both are used, {mat_a} must be adjacent to {mat_b}. ")
            constraints.append({"type": "adjacency", "material_a": mat_a,
                                "material_b": mat_b, "must_be_adjacent": True})
        else:
            texts.append(f"If both are used, {mat_a} cannot be adjacent to {mat_b}. ")
            constraints.append({"type": "adjacency", "material_a": mat_a,
                                "material_b": mat_b, "must_be_adjacent": False})

    # Relative thickness constraints
    n_rel = int(np_rng.poisson(avg_rel_tkns_constraints))
    for _ in range(n_rel):
        if len(all_mats) < 2:
            break
        idx_a, idx_b = rng.sample(range(len(all_mats)), 2)
        mat_a, mat_b = all_mats[idx_a], all_mats[idx_b]
        t_a, t_b = all_thks[idx_a], all_thks[idx_b]
        if t_a > t_b:
            texts.append(f"If both are used, {mat_a} must be thicker than {mat_b}. ")
            constraints.append({"type": "relative_thickness", "material_a": mat_a,
                                "material_b": mat_b, "relation": "thicker"})
        elif t_a < t_b:
            texts.append(f"If both are used, {mat_a} must be thinner than {mat_b}. ")
            constraints.append({"type": "relative_thickness", "material_a": mat_a,
                                "material_b": mat_b, "relation": "thinner"})
        else:
            texts.append(f"If both are used, {mat_a} must have an equal thickness to {mat_b}. ")
            constraints.append({"type": "relative_thickness", "material_a": mat_a,
                                "material_b": mat_b, "relation": "equal"})

    # Individual thickness constraints
    n_indv = int(np_rng.poisson(avg_indv_tkns_constraints))
    for _ in range(n_indv):
        options = ["Thicker", "Thinner", "Exact"]
        weights = [0.45, 0.45, 0.1]
        choice = rng.choices(options, weights=weights, k=1)[0]
        idx = rng.randrange(len(all_mats))
        mat = all_mats[idx]
        thickness = all_thks[idx]
        noise = int(np_rng.poisson(thickness * 0.25 / 5)) * 5 + 5

        if choice == "Thicker":
            minimum = max(thickness - noise, 5)
            texts.append(f"If used, {mat} must be thicker than {minimum} nm. ")
            constraints.append({"type": "individual_thickness", "material": mat,
                                "bound": "min", "value_nm": minimum})
        elif choice == "Thinner":
            maximum = thickness + noise
            texts.append(f"If used, {mat} must be thinner than {maximum} nm. ")
            constraints.append({"type": "individual_thickness", "material": mat,
                                "bound": "max", "value_nm": maximum})
        else:
            texts.append(f"If used, {mat} must be exactly {thickness} nm thick. ")
            constraints.append({"type": "individual_thickness", "material": mat,
                                "bound": "exact", "value_nm": thickness})

    # Total thickness constraint
    if rng.random() <= ttl_tkns_constraint_prob:
        sub_options = ["Sum", "Each layer"]
        sub_weights = [0.2, 0.8]
        sub_choice = rng.choices(sub_options, weights=sub_weights, k=1)[0]

        if sub_choice == "Sum":
            noise = int(np_rng.poisson(sum(thicknesses) * 0.25 / 5)) * 5 + 5
            bound_opts = ["Maximum", "Minimum"]
            bound_choice = rng.choices(bound_opts, weights=[0.5, 0.5], k=1)[0]
            if bound_choice == "Maximum":
                maximum = sum(thicknesses) + noise
                texts.append(f"The total thickness of the structure must be less than {maximum} nm. ")
                constraints.append({"type": "total_thickness_sum", "bound": "max",
                                    "value_nm": maximum})
            else:
                minimum = max(sum(thicknesses) - noise, 10)
                texts.append(f"The total thickness of the structure must be greater than {minimum} nm. ")
                constraints.append({"type": "total_thickness_sum", "bound": "min",
                                    "value_nm": minimum})
        else:
            bound_opts = ["Maximum", "Minimum"]
            bound_choice = rng.choices(bound_opts, weights=[0.5, 0.5], k=1)[0]
            if bound_choice == "Maximum":
                noise = int(np_rng.poisson(max(thicknesses) * 0.25 / 5)) * 5 + 5
                maximum = max(thicknesses) + noise
                texts.append(f"Each of the layers in the structure must be thinner than {maximum} nm. ")
                constraints.append({"type": "total_thickness_each", "bound": "max",
                                    "value_nm": maximum})
            else:
                noise = int(np_rng.poisson(min(thicknesses) * 0.25 / 5)) * 5 + 5
                minimum = max(min(thicknesses) - noise, 5)
                texts.append(f"Each of the layers in the structure must be thicker than {minimum} nm. ")
                constraints.append({"type": "total_thickness_each", "bound": "min",
                                    "value_nm": minimum})

    # Layer identity constraint
    if rng.random() <= lyr_id_constraint_prob:
        pos_opts = ["First", "Last", "Both"]
        pos_weights = [0.35, 0.35, 0.3]
        pos_choice = rng.choices(pos_opts, weights=pos_weights, k=1)[0]

        not_first = [m for m in all_mats if m != materials[0]]
        not_last = [m for m in all_mats if m != materials[-1]]
        rand_not_first = rng.choice(not_first) if not_first else materials[0]
        rand_not_last = rng.choice(not_last) if not_last else materials[-1]

        if pos_choice == "First":
            is_not = rng.choices(["Is", "Not"], weights=[0.5, 0.5], k=1)[0]
            if is_not == "Is":
                texts.append(f"The first layer in the stack must be {materials[0]}. ")
                constraints.append({"type": "layer_identity", "position": "first",
                                    "material": materials[0], "must_be": True})
            else:
                texts.append(f"The first layer in the stack must not be {rand_not_first}. ")
                constraints.append({"type": "layer_identity", "position": "first",
                                    "material": rand_not_first, "must_be": False})
        elif pos_choice == "Last":
            is_not = rng.choices(["Is", "Not"], weights=[0.5, 0.5], k=1)[0]
            if is_not == "Is":
                texts.append(f"The last layer in the stack must be {materials[-1]}. ")
                constraints.append({"type": "layer_identity", "position": "last",
                                    "material": materials[-1], "must_be": True})
            else:
                texts.append(f"The last layer in the stack must not be {rand_not_last}. ")
                constraints.append({"type": "layer_identity", "position": "last",
                                    "material": rand_not_last, "must_be": False})
        else:  # Both
            sub = rng.choices(["Is-Is", "Is-Not", "Not-Is", "Not-Not"],
                              weights=[0.25]*4, k=1)[0]
            if sub == "Is-Is":
                texts.append(f"The first layer in the stack must be {materials[0]} "
                             f"and the last layer must be {materials[-1]}. ")
                constraints.append({"type": "layer_identity", "position": "first",
                                    "material": materials[0], "must_be": True})
                constraints.append({"type": "layer_identity", "position": "last",
                                    "material": materials[-1], "must_be": True})
            elif sub == "Is-Not":
                texts.append(f"The first layer in the stack must be {materials[0]} "
                             f"and the last layer must not be {rand_not_last}. ")
                constraints.append({"type": "layer_identity", "position": "first",
                                    "material": materials[0], "must_be": True})
                constraints.append({"type": "layer_identity", "position": "last",
                                    "material": rand_not_last, "must_be": False})
            elif sub == "Not-Is":
                texts.append(f"The first layer in the stack must not be {rand_not_first} "
                             f"and the last layer must be {materials[-1]}. ")
                constraints.append({"type": "layer_identity", "position": "first",
                                    "material": rand_not_first, "must_be": False})
                constraints.append({"type": "layer_identity", "position": "last",
                                    "material": materials[-1], "must_be": True})
            else:  # Not-Not
                texts.append(f"The first layer in the stack must not be {rand_not_first} "
                             f"and the last layer must not be {rand_not_last}. ")
                constraints.append({"type": "layer_identity", "position": "first",
                                    "material": rand_not_first, "must_be": False})
                constraints.append({"type": "layer_identity", "position": "last",
                                    "material": rand_not_last, "must_be": False})

    return texts, constraints
