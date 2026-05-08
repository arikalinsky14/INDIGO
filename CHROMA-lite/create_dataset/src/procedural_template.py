import numpy as np
import random as _random
_rng = _random.Random()
_np_rng = np.random.default_rng()
import copy
from color_palette_manager import PaletteMatcher as PM

PM.register_palette("common", {
    # Neutrals
    "Black":       "#000000",
    "White":       "#FFFFFF",
    "Gray":        "#808080",
    "Light Gray":  "#D3D3D3",
    "Dark Gray":   "#A9A9A9",
    "Silver": "#C0C0C0",      # bright metallic gray, cooler tone
    "Charcoal": "#36454F",    # dark cool gray, distinct from 'Dark Gray'
    "Ivory": "#FFFFF0",       # off-white with a yellow tint

    # Reds & Pinks
    "Red":         "#FF0000",
    "Dark Red":    "#8B0000",
    "Maroon":      "#800000",
    "Pink":        "#FFC0CB",
    "Hot Pink":    "#FF69B4",
    "Salmon":      "#FA8072",

    # Oranges & Yellows
    "Orange":      "#FFA500",
    "Dark Orange": "#FF8C00",
    "Gold":        "#FFD700",
    "Yellow":      "#FFFF00",
    "Peach": "#FFDAB9",       # light orange-pink
    "Mustard": "#FFDB58",     # muted, earthy yellow
    "Lemon": "#FFF44F",       # lighter yellow
    "Pitt Gold": "#FFB81C",

    # Greens

    "Green":       "#008000",
    "Dark Green":  "#006400",
    "Light Green": "#90EE90",
    "Lime":        "#00FF00",
    "Olive":       "#3C4C24",
    "Forest Green": "#228B22",  # deep natural green
    "Sage": "#9DC183",          # muted gray-green
    "Mint": "#98FF98",          # soft pastel green
    "Pale Mint": "#CCF3D8",
    

    # Blues
    "Blue": "#1E40FF",           # vivid but balanced blue
    "Light Blue": "#7EC8E3",     # realistic light sky blue
    "Navy": "#001F3F",           # authentic deep navy
    "Teal": "#008C8C",           # balanced blue-green teal
    "Cyan": "#00BCD4",           # print/process cyan
    "Turquoise": "#30D5C8",      # natural turquoise stone hue
    "Steel Blue": "#4682B4", "Cornflower Blue": "#6495ED", "Dodger Blue": "#1E90FF",
    "Midnight Blue": "#191970",   # very dark, slightly violet blue
    "Denim": "#1560BD",           # muted medium blue
    "Cadet Blue": "#5F9EA0",      # grayish teal-blue
    "Slate Blue": "#6A5ACD",      # muted violet-blue
    "Deep Sea Blue": "#08457E",   # rich dark ocean blue
    "Dark Cyan": "#008B8B",       # deep cyan-green
    "Storm Blue": "#274060",
    "Pitt Royal Blue": "#003594",

    # Purples & Magentas
    "Purple":          "#800080",  # classic purple
    "Indigo":          "#4B0082",  # deep blue-purple
    "Magenta":         "#FF00FF",  # vivid pink-purple
    "Violet":          "#EE82EE",  # soft light purple
    "Orchid":          "#DA70D6",  # mid lavender-pink
    "Medium Purple":   "#9370DB",  # balanced purple tone
    "Blue Violet":     "#8A2BE2",  # strong blue-leaning purple
    "Plum":            "#DDA0DD",  # desaturated pale purple
    "Wine": "#722F37",       # dark reddish-purple
    "Eggplant":      "#614051",  # dark, muted purple with gray undertone
    "Grape":         "#6F2DA8",  # rich, medium-dark purple
    "Dark Orchid":   "#9932CC",  # vivid deep purple (brightest of the set)
    "Deep Violet":   "#9400D3",  # true deep violet
    "Amethyst":      "#9966CC",  # softer, jewel-toned purple


    # Browns / Earth Tones
    "Brown":       "#4D2E2E",
    "Light Brown":   "#BB4E00",
    "Tan":         "#E0CAAC",
    "Beige": "#F5F5DC",
    "Khaki": "#F0E68C",
    "Chocolate": "#7B3F00",   # darker warm brown
    "Coffee": "#6F4E37",      # earthy neutral brown
    "Rust": "#B7410E",        # reddish-brown
    "Sand": "#C2B280",        # light neutral tan
})
PM.activate("common")


all_materials = [
            'Ag',
            'Al',
            'Al2O3',
            'Au',
            'AZO',
            'Cr',
            'GaAs',
            'GaInP',
            'GaP',
            'Ge',
            'InP',
            'ITO',
            'Mn',
            'Ni',
            'Pd',
            'Pt',
            'Si3N4',
            'SiO2',
            'Ti',
            'TiN',
            'TiO2',
            'aSi',
            'cSi',
            'W',
            'ZnO'
        ]

def get_str_materials(materials):
    materials_copied = copy.deepcopy(materials)
    _np_rng.shuffle(materials_copied)

    materials_copied = list(materials_copied)
    return ', '.join(materials_copied)

def get_unused_materials(materials):
    return [item for item in all_materials if item not in materials]

def get_str_a_num_layers(num_layers):
    if num_layers == 1:
        str_num_layers = 'a one'
    elif num_layers == 2:
        str_num_layers = 'a two'
    elif num_layers == 3:
        str_num_layers = 'a three'
    elif num_layers == 4:
        str_num_layers = 'a four'
    elif num_layers == 5:
        str_num_layers = 'a five'
    elif num_layers == 6:
        str_num_layers = 'a six'
    elif num_layers == 7:
        str_num_layers = 'a seven'
    elif num_layers == 8:
        str_num_layers = 'an eight'
    elif num_layers == 9:
        str_num_layers = 'a nine'
    elif num_layers == 10:
        str_num_layers = 'a ten'
    else:
        raise ValueError

    return str_num_layers

def get_str_num_layers(num_layers):
    if num_layers == 1:
        str_num_layers = 'one'
    elif num_layers == 2:
        str_num_layers = 'two'
    elif num_layers == 3:
        str_num_layers = 'three'
    elif num_layers == 4:
        str_num_layers = 'four'
    elif num_layers == 5:
        str_num_layers = 'five'
    elif num_layers == 6:
        str_num_layers = 'six'
    elif num_layers == 7:
        str_num_layers = 'seven'
    elif num_layers == 8:
        str_num_layers = 'eight'
    elif num_layers == 9:
        str_num_layers = 'nine'
    elif num_layers == 10:
        str_num_layers = 'ten'
    else:
        raise ValueError

    return str_num_layers

def get_layers_fragment(num_layers,specify=False):
    if specify:
        return f"Create an optical structure that consists of {get_str_a_num_layers(num_layers)}-layer stack; "

    noise: int =_np_rng.poisson(num_layers*.25)
    options = ["Min", "Max", "Exact", "None"]
    weights = [0.125, 0.125, 0.25, 0.50]
    choice = _rng.choices(options, weights=weights, k=1)[0]

    if(choice == "Min"):
        minimum = max(num_layers - noise,2)
        return f"Create an optical structure that consists of a stack with at least {get_str_num_layers(minimum)} layers; ", choice
    elif(choice == "Max"):
        maximum = min(num_layers + noise,10)
        return f"Create an optical structure that consists of a stack with at most {get_str_num_layers(maximum)} layers; ", choice
    elif(choice == "Exact"):
        return f"Create an optical structure that consists of {get_str_a_num_layers(num_layers)}-layer stack; ", choice
    else:
        return f"Create an optical structure that consists of a layered stack; ", choice
    
def get_materials_fragment(materials,specify=False):
    if specify:
        return f"the stack must be composed of {get_str_materials(materials)}. ", []
    
    
    options = ["Strict", "Helpful", "Extra", "Restrict", "Any"]
    weights = [0.05, 0.05, 0.30, 0.30, 0.30]
    choice = _rng.choices(options, weights=weights, k=1)[0]

    if(choice == "Strict"):
        return f"the stack must be composed of {get_str_materials(materials)}. ", []
    elif(choice == "Helpful"):
        return f"the stack may be composed of {get_str_materials(materials)}, but usage of all materials is not required. ", []
    elif(choice == "Extra"):
        extra_choices: int = min(_np_rng.poisson(len(get_unused_materials(materials))*.2),9) + 1 
        extra_materials =_rng.sample(get_unused_materials(materials),extra_choices)
        new_materials = materials + extra_materials
        return f"the stack may be composed of {get_str_materials(new_materials)}, but usage of all materials is not required. ", extra_materials
    elif(choice == "Restrict"):
        unused_materials = get_unused_materials(materials)
        restrict_count: int = max(min(_np_rng.poisson(len(unused_materials) * 0.3), len(unused_materials)),1)
        if restrict_count > 0:
            restricted_materials =_rng.sample(unused_materials, restrict_count)
            remaining_materials = [mat for mat in unused_materials if mat not in restricted_materials]
            return f"the stack may be composed of any material except {get_str_materials(restricted_materials)}. ", remaining_materials
    else:
        extra_materials = get_unused_materials(materials)
        return f"the stack may be composed of any material. ", extra_materials

def get_color_identity(color_rgb: tuple[int, int, int]) -> str:
    options = ["RGB", "Name", "Specific Name", "HEX"]
    weights = [1/3, 1/6, 1/6, 1/3]
    choice =_rng.choices(options, weights=weights, k=1)[0]
    r, g, b = map(int, color_rgb)
    
    if choice == "RGB":
        return f"The reflected color observed from this multilayer configuration must correspond to the RGB value ({r}, {g}, {b}). "
    elif choice == "Name":
        common_name = PM.find(r, g, b)
        return f"The reflected color observed from this multilayer configuration must correspond to {common_name}. "
    elif choice == "Specific Name":
        with PM.use("__full__"):
            full_name = PM.find(r, g, b)
        return f"The reflected color observed from this multilayer configuration must correspond to {full_name}. "
    elif choice == "HEX":
        hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return f"The reflected color observed from this multilayer configuration must correspond to the HEX code {hex_code}. "


def additional_constraints(
        materials, 
        thicknesses, 
        extra_materials, 
        avg_adj_constraints = 0.025, #average adjacent constraint count
        avg_rel_tkns_constraints = 0.025, #average relative thickness constraint count
        avg_indv_tkns_constraints =0.8, #average indivudal thickness constraint count
        ttl_tkns_constraint_prob = 0.55, #total thickness constraint probability
        lyr_id_constaint_prob = 0.02, #layer id constraint probability
        force_zero = False
    ):

    # Create combined lists for uniform selection
    all_materials_for_constraints = materials + extra_materials
    all_thicknesses_for_constraints = thicknesses + [
        # Generate thickness for each extra material based on random actual material thickness + noise
        max((thicknesses[_rng.randrange(len(thicknesses))] + 
             (_np_rng.poisson(thicknesses[_rng.randrange(len(thicknesses))] * 0.25 / 5) * 5 + 5) *_rng.choice([-1, 1])), 
            5)  # Ensure multiple of 5 and minimum 5
        for _ in extra_materials
    ]
    
    #Define additional constraint types
    def adjacency(all_materials_for_constraints, materials):
        # Select two materials uniformly from all available materials
        material_a, material_b =_rng.sample(all_materials_for_constraints, 2)
        
        # Check if both materials are in the actual structure
        material_a_in_structure = material_a in materials
        material_b_in_structure = material_b in materials
        
        # If both are in structure, use original logic
        if (material_a_in_structure and material_b_in_structure):
            index_a = materials.index(material_a)
            index_b = materials.index(material_b)
            
            if(index_a + 1 == index_b or index_a - 1 == index_b):
                return f"If both are used, {material_a} must be adjacent to {material_b}. "
            else:
                return f"If both are used, {material_a} cannot be adjacent to {material_b}. "
        else:
            index_a = all_materials_for_constraints.index(material_a)
            index_b = all_materials_for_constraints.index(material_b)

            if(index_a + 1 == index_b or index_a - 1 == index_b):
                return f"If both are used, {material_a} must be adjacent to {material_b}. "
            else:
                return f"If both are used, {material_a} cannot be adjacent to {material_b}. "
        
    def relative_thickness(all_materials_for_constraints, all_thicknesses_for_constraints):
        # Select two materials uniformly from all available materials
        index_a, index_b =_rng.sample(range(len(all_materials_for_constraints)), 2)
        material_a, material_b = all_materials_for_constraints[index_a], all_materials_for_constraints[index_b]
        thickness_a, thickness_b = all_thicknesses_for_constraints[index_a], all_thicknesses_for_constraints[index_b]

        if(thickness_a > thickness_b):
            return f"If both are used, {material_a} must be thicker than {material_b}. "
        elif(thickness_a < thickness_b):
            return f"If both are used, {material_a} must be thinner than {material_b}. "
        else:
            return f"If both are used, {material_a} must have an equal thickness to {material_b}. "
        
    def individual_thickness(all_materials_for_constraints, all_thicknesses_for_constraints):
        options = ["Thicker", "Thinner", "Exact"]
        weights = [0.45, 0.45, 0.1]
        choice =_rng.choices(options, weights=weights, k=1)[0]
        index =_rng.randrange(len(all_materials_for_constraints))

        material = all_materials_for_constraints[index]
        thickness = all_thicknesses_for_constraints[index]
        noise: int =_np_rng.poisson(thickness * 0.25 / 5) * 5 + 5

        if(choice == "Thicker"):
            minimum = max(thickness - noise, 5) 
            return f"If used, {material} must be thicker than {minimum} nm. "
        elif(choice == "Thinner"):
            maximum = thickness + noise
            return f"If used, {material} must be thinner than {maximum} nm. "
        else:
            return f"If used, {material} must be exactly {thickness} nm thick. "
        
    def total_thickness(thicknesses):
        options = ["Sum", "Each layer"]
        weights = [0.2, 0.8]
        choice =_rng.choices(options, weights=weights, k=1)[0]

        if(choice == "Sum"):
            noise: int =_np_rng.poisson(sum(thicknesses)*.25/5)*5 + 5
            options = ["Maximum", "Minimum"]
            weights = [0.5, 0.5]
            choice_2 =_rng.choices(options, weights=weights, k=1)[0]

            if(choice_2 == "Maximum"):
                maximum = sum(thicknesses) + noise
                return f"The total thickness of the structure must be less than {maximum} nm. "
            elif(choice_2 == "Minimum"):
                minimum = max(sum(thicknesses) - noise,10)
                return f"The total thickness of the structure must be greater than {minimum} nm. "
        elif(choice == "Each layer"):
            options = ["Maximum", "Minimum"]
            weights = [0.5, 0.5]
            choice_2 =_rng.choices(options, weights=weights, k=1)[0]

            if(choice_2 == "Maximum"):
                noise: int =_np_rng.poisson(max(thicknesses)*.25/5)*5 + 5

                maximum = max(thicknesses) + noise
                return f"Each of the layers in the structure must be thinner than {maximum} nm. "
            elif(choice_2 == "Minimum"):
                noise: int =_np_rng.poisson(min(thicknesses)*.25/5)*5 + 5

                minimum = max(min(thicknesses) - noise,5)
                return f"Each of the layers in the structure must be thicker than {minimum} nm. "
            
    def layer_identity(all_materials_for_constraints, materials):
        options = ["First", "Last", "Both"]
        weights = [0.35, 0.35, .3]
        choice =_rng.choices(options, weights=weights, k=1)[0]

        not_first_materials = [mat for mat in all_materials_for_constraints if mat != materials[0]]
        random_not_first_material =_rng.choice(not_first_materials)
        not_last_materials = [mat for mat in all_materials_for_constraints if mat != materials[-1]]
        random_not_last_material =_rng.choice(not_last_materials)
        
        if(choice == 'First'):
            options = ["Is", "Not"]
            weights = [0.5, 0.5]
            choice_2 =_rng.choices(options, weights=weights, k=1)[0]
            if(choice_2 == "Is"):
                return f"The first layer in the stack must be {materials[0]}. "
            else:
                return f"The first layer in the stack must not be {random_not_first_material}. "
        elif(choice == 'Last'):
            options = ["Is", "Not"]
            weights = [0.5, 0.5]
            choice_2 =_rng.choices(options, weights=weights, k=1)[0]
            if(choice_2 == "Is"):
                return f"The last layer in the stack must be {materials[-1]}. "
            else:
                return f"The last layer in the stack must not be {random_not_last_material}. "
        elif(choice == 'Both'):
            # 4 subconditions: Is-Is, Is-Not, Not-Is, Not-Not
            options = ["Is-Is", "Is-Not", "Not-Is", "Not-Not"]
            weights = [0.25, 0.25, 0.25, 0.25]
            choice_2 =_rng.choices(options, weights=weights, k=1)[0]
            
            if(choice_2 == "Is-Is"):
                return f"The first layer in the stack must be {materials[0]} and the last layer must be {materials[-1]}. "
            elif(choice_2 == "Is-Not"):
                return f"The first layer in the stack must be {materials[0]} and the last layer must not be {random_not_last_material}. "
            elif(choice_2 == "Not-Is"):
                return f"The first layer in the stack must not be {random_not_first_material} and the last layer must be {materials[-1]}. "
            elif(choice_2 == "Not-Not"):
                return f"The first layer in the stack must not be {random_not_first_material} and the last layer must not be {random_not_last_material}. "
    
    # Use defined constraint averages to determine the number of constraints
    adj_constraints: int =_np_rng.poisson(avg_adj_constraints)
    rel_tkns_constraints: int =_np_rng.poisson(avg_rel_tkns_constraints)
    indv_tkns_constraints: int =_np_rng.poisson(avg_indv_tkns_constraints)
    ttl_tkns_constraint =_rng.random() <= ttl_tkns_constraint_prob
    layer_id_constraint =_rng.random() <= lyr_id_constaint_prob
    if force_zero:
        adj_constraints = 0
        rel_tkns_constraints = 0
        indv_tkns_constraints = 0
        ttl_tkns_constraint = False
        layer_id_constraint = False
    constraints = []
    
    #Add constraints
    for _ in range(adj_constraints):
        constraints.append(adjacency(all_materials_for_constraints, materials))
    for _ in range(rel_tkns_constraints):
        constraints.append(relative_thickness(all_materials_for_constraints, all_thicknesses_for_constraints))
    for _ in range (indv_tkns_constraints):
        constraints.append(individual_thickness(all_materials_for_constraints, all_thicknesses_for_constraints))
    if(ttl_tkns_constraint):
        constraints.append(total_thickness(thicknesses))
    if(layer_id_constraint):
        constraints.append(layer_identity(all_materials_for_constraints, materials))

    # Join constraints into a single string
    return constraints

def get_template(num_layers, materials, thicknesses, incidence_angle, color_rgb, picked_seed = 42, verbose = False):

    _rng.seed(picked_seed)
    global _np_rng
    _np_rng = np.random.default_rng(picked_seed)
    procedural_template, _ = get_layers_fragment(num_layers)
    procedural_materials, extra_materials = get_materials_fragment(materials)
    procedural_template += procedural_materials
    #procedural_template += f"At {incidence_angle} degrees, "
    procedural_template += get_color_identity(color_rgb)
    additional_constraints_list = additional_constraints(materials, thicknesses, extra_materials)
    _random.shuffle(additional_constraints_list)
    procedural_template += "".join(additional_constraints_list)

    if verbose:
        print(f"extra materials {extra_materials}")
        print('')
    return procedural_template

if __name__ == '__main__':

    num_layers = 4
    materials = ["SiO2", "TiO2", "Si3N4", "InP"]
    thicknesses = [30, 175, 185, 120]
    incidence_angle = 0
    color_rgb = [186, 80, 80]


    print('=' * 50)
    print(f'num_layers {num_layers}')
    print(f'materials {materials}')
    print(f'thicknesses {thicknesses}')
    print(f'incidence_angle {incidence_angle}')
    print(f'color_rgb {color_rgb}')
    
    #global _np_rng
    for sed in range(20):
        prompt_seed = sed
        print(get_template(num_layers, materials, thicknesses, incidence_angle, color_rgb, picked_seed=prompt_seed, verbose=True))
        print("=" * 60)
