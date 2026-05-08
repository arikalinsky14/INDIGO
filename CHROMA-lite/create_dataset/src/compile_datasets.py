import argparse
import os
import pandas as pd
import numpy as onp
import jax.numpy as jnp
import scipy.constants as scic
import time
import random as random
import errno


import jaxlayerlumos.colors.composite as jll_colors_composite

import procedural_template
from rephraser import Rephraser 



def parse_materials(str_materials):
    list_str_materials = str_materials.split('_')
    new_str_materials = []

    for str_material in list_str_materials:
        if str_material == 'AZO-Zarei':
            str_material = 'AZO'

        new_str_materials.append(str_material)

    return new_str_materials


def parse_thicknesses(str_thicknesses):
    list_str_thicknesses = str_thicknesses.split('_')
    list_str_thicknesses = [int(str_thickness[:-2]) for str_thickness in list_str_thicknesses]

    return list_str_thicknesses


"""def merge_materials_thicknesses(str_materials, str_thicknesses):
    list_str_materials = parse_materials(str_materials)
    list_str_thicknesses = parse_thicknesses(str_thicknesses)

    list_str_materials_thicknesses = []
    for material, thickness in zip(list_str_materials, list_str_thicknesses):
        list_str_materials_thicknesses.append(f'{material}_{thickness:d}')

    return list_str_materials_thicknesses"""


def transform_to_sRGB(wavelengths_in_nm, spectrum):
    assert wavelengths_in_nm.shape[0] == spectrum.shape[0]

    indices = onp.logical_and(wavelengths_in_nm > 360, wavelengths_in_nm < 830)

    sRGB = jll_colors_composite.spectrum_to_sRGB(
        jnp.array(wavelengths_in_nm[indices]),
        jnp.array(spectrum[indices]),
        use_clipping=True
    )

    sRGB = onp.array(sRGB)
    sRGB *= 255
    #print("Original sRGB:", sRGB)

    sRGB = sRGB.flatten()[:3]
    sRGB = onp.round(sRGB).astype(int).tolist()
    #print("New sRGB:", sRGB)
    return sRGB


def get_input_dir(num_layers, incidence_angle, seed):
    input_dir = 'data_struct'
    input_dir = os.path.join(input_dir, f'layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi')
    input_dir = os.path.join(input_dir, f"TR_simulations_layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi_seed_{seed:05d}.parquet")

    return input_dir

def get_output_dir(num_layers, incidence_angle, seed):
    output_dir = 'data_prompts'
    output_dir = os.path.join(output_dir, f'layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi')
    output_dir = os.path.join(output_dir, f"TR_simulations_layers_{num_layers:02d}_angle_{incidence_angle:02d}_substrate_CSi_seed_{seed:05d}.parquet")
    return output_dir


def load_structures(input_dir):
    df = pd.read_parquet(input_dir)
    print(df.shape)

    structures = []
    num_structures = 10000

    for ind_structure in range(0, num_structures):
        df_structure = df[df['Index'] == ind_structure]
        structure = df_structure.to_numpy()
        #assert all thicknesses and material columns are the same (since same structure)
        assert onp.all(structure[:, 1] == structure[0, 1])
        assert onp.all(structure[:, 2] == structure[0, 2])

        #append material, thickness... all numeric data omward [wavelength, R, T]
        structures.append([structure[0, 1], structure[0, 2], structure[:, 3:].astype(onp.float32)])

    return structures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--incidence_angle', type=int, required=True)
    parser.add_argument('--structure_seed', type=int, required=True)

    args = parser.parse_args()

    num_layers = args.num_layers
    incidence_angle = args.incidence_angle
    structure_seed = args.structure_seed
    pid = os.getpid()
    prompt_seed_generator = random.Random(structure_seed)


    input_dir = get_input_dir(num_layers, incidence_angle, structure_seed)
    structures = load_structures(input_dir)

    #Load an instance of the Open-AI called class
    rephrasingHelper = Rephraser(model="gpt-5-mini-2025-10-01")

    rows = []
    for structure in structures:
        materials = parse_materials(structure[0])
        thicknesses = parse_thicknesses(structure[1])
        assert len(materials) == len(thicknesses)

        num_layers_from_struct = len(materials) 
        assert num_layers_from_struct == num_layers

        wavelengths = structure[2][:, 0]
        wavelengths_in_nm = wavelengths / scic.nano
        R = structure[2][:, 1]
        T = structure[2][:, 2]

        sRGB_R = transform_to_sRGB(wavelengths_in_nm, R)
        random_prompt_seed = prompt_seed_generator.randint(0, 2**32 - 1)

        str_prompt = procedural_template.get_template(
            num_layers_from_struct, materials, thicknesses, incidence_angle, sRGB_R, random_prompt_seed, verbose=False
        )
        
        rephrased = rephrasingHelper.rephrase(text=str_prompt,seed=random_prompt_seed)

        row = {
            "structure_seed": structure_seed,
            "prompt_seed": random_prompt_seed,
            "num_layers": num_layers_from_struct,
            "materials": ",".join(materials),
            "thicknesses": ",".join(str(t) for t in thicknesses),
            "incidence_angle": incidence_angle,
            "sRGB_R": sRGB_R,
            "prompt_template": str_prompt,
            "dev_prompt": rephrased.chosen_prompt_name,
            "rephrased_prompt": rephrased.text

        }
        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_parquet(get_output_dir(num_layers,incidence_angle,seed=structure_seed), index=False)


