import numpy as np
import random as _random
_rng = _random.Random()
_np_rng = np.random.default_rng()
import procedural_template as template_maker
import math


all_materials = template_maker.all_materials
invalid_materials = [
    # --- (kept from your list) transition & other metals ---
    'Cu', 'V', 'Nb', 'Ta', 'Zr', 'Hf', 'Mo', 'Re', 'Os', 'Ir', 'Co', 'Fe',
    'Mg', 'Li', 'Na', 'Ca', 'Sr', 'Ba',
    'Sb', 'Bi', 'Sn', 'Pb', 'Te',
    'Ce', 'Y', 'La', 'Gd', 'Er', 'Dy',
    'NiCr', 'TiAlN', 'ZrN', 'HfN', 'TaN', 'MoN', 'WN', 'VN', 'CuAlO2', 'MgAl2O4',
    'Ru', 'Sc', 'Yb', 'Th', 'U',

    # --- additional elements (exclude overlaps with valid) ---
    'B', 'C', 'P', 'S', 'Se', 'Po', 'As',
    'Be', 'K', 'Rb', 'Cs',
    'TiAl',  # distinct from Ti or TiN (valid)
    'Ga', 'In', 'Tl',
    'Hg', 'Cd', 'Zn',  # note: ZnO is valid; elemental Zn excluded
    'Pr', 'Nd', 'Sm', 'Eu', 'Tb', 'Ho', 'Tm', 'Lu',

    # --- simple salts / halides / fluorides ---
    'NaCl', 'KCl', 'LiCl', 'RbCl', 'CsCl',
    'NaF', 'KF', 'LiF', 'RbF', 'CsF',
    'MgF2', 'CaF2', 'SrF2', 'BaF2',
    'AlF3', 'ZnF2', 'ZrF4', 'HfF4',

    # --- oxides (avoid Al2O3, SiO2, ZnO, ITO which are valid) ---
    'Cu2O', 'CuO',
    'FeO', 'Fe2O3', 'Fe3O4',
    'CoO', 'Co3O4',
    'NiO',  # Ni is valid, but NiO excluded
    'V2O5', 'VO2',
    'Nb2O5', 'Ta2O5',
    'ZrO2', 'HfO2',
    'MoO3', 'WO3',  # W (metal) is valid; WO3 excluded
    'SnO', 'SnO2',
    'Ga2O3', 'In2O3',
    'Ti2O3', 'TiO',  # TiO2 is valid; other Ti oxides excluded
    'Cr2O3',  # Cr metal valid; oxide excluded here
    'SiO',    # SiO2 is valid; sub-oxide excluded
    'Y2O3', 'La2O3', 'CeO2',
    'ZnAl2O4',  # gahnite spinel
    'CoAl2O4',  # cobalt aluminate spinel
    'SrTiO3',   # perovskite oxide
    'BaTiO3',   # ferroelectric perovskite

    # --- nitrides & carbides (avoid TiN which is valid) ---
    'AlN', 'GaN', 'InN',
    'BN', 'cBN',
    'SiON',
    'SiC', 'BCN',
    'TiC', 'ZrC', 'HfC', 'VC', 'NbC', 'TaC', 'Mo2C', 'WC',
    'CrN', 'FeN', 'CoN', 'NiN',
    'GaCN', 'AlON',  # oxynitrides

    # --- chalcogenides (sulfides/selenides/tellurides) ---
    'MoS2', 'WS2', 'MoSe2', 'WSe2',
    'FeS2', 'CuS', 'Cu2S', 'CuSe', 'Cu2Se',
    'ZnS', 'ZnSe', 'ZnTe',  # note: ZnO is valid; ZnS/Se/Te excluded
    'CdS', 'CdSe', 'CdTe',
    'PbS', 'PbSe', 'PbTe',
    'In2Se3', 'Ga2Se3',
    'Bi2Se3', 'Bi2Te3',
    'SnS', 'SnSe',

    # --- III–V & II–VI semiconductors not in valid list ---
    'InAs', 'GaSb', 'InSb', 'AlAs', 'AlP', 'InGaAs', 'InGaN', 'AlGaN',
    'GaInAsP', 'GaNAs', 'GaPN',

    # --- perovskites & halide perovskites ---
    'MAPbI3', 'MAPbBr3', 'FAPbI3', 'CsPbI3', 'CsPbBr3', 'CsSnI3',
    'Rb2SnI6', 'BaSnO3', 'SrSnO3',

    # --- other ceramics & glasses ---
    'SiOxNy', 'SiOC',  
    'B2O3', 'P2O5', 'GeO2',        # glass formers (Ge metal is valid; oxide excluded)
    'LiAlSiO4', 'NaAlSi3O8',       # feldspar-like aluminosilicates
    'Cordierite', 'Mullite',
    'BaZrO3', 'LaAlO3', 'YAG', 'YSZ',

    # --- polymers / organics (common thin-film resists & polymers) ---
    'PMMA', 'SU-8', 'PDMS', 'Parylene-C', 'PI', 'PEI', 'PVP', 'PVA',
    'PC', 'PS', 'PET', 'PMGI', 'CYTOP', 'Teflon', 'PTFE', 'FEP',

    # --- liquids / solvents (to exclude accidental picks) ---
    'H2O', 'D2O', 'Ethanol', 'Isopropanol', 'Acetone', 'Toluene',
    'Glycerol', 'MineralOil', 'SiliconeOil',

    # --- gases / plasmas (non-solid films in this context) ---
    'Air', 'N2', 'O2', 'Ar', 'He', 'Ne', 'Kr', 'Xe',

    # --- transparent conductors not in valid list (keep ITO/AZO valid) ---
    'FTO',  # fluorine-doped SnO2

    # --- misc. mixed / intermetallics (distinct from valid metals) ---
    'CuZn', 'CuNi', 'FeCr', 'CoFe', 'NiFe', 'AlCu', 'AlMg',
    'GaIn', 'InSn', 'SnBi', 'PbSn',
]

all_and_invalid_materials = all_materials + invalid_materials



def get_errored_materials_fragment(materials):
    
    
    options = ["Strict", "Helpful", "Extra", "Restrict"]
    weights = [0.05, 0.05, 0.40, 0.40]
    choice = _rng.choices(options, weights=weights, k=1)[0]
    invalid_choices: int = max(min(_np_rng.poisson(len(template_maker.get_unused_materials(materials))*.25),15),1)
    chosen_invalid_materials =_rng.sample(invalid_materials,invalid_choices)
    materials_with_invalids = chosen_invalid_materials + materials

    if(choice == "Strict"):
        return f"the stack must be composed of {template_maker.get_str_materials(materials_with_invalids)}. ", chosen_invalid_materials, chosen_invalid_materials
    elif(choice == "Helpful"):
        return f"the stack may be composed of {template_maker.get_str_materials(materials_with_invalids)}, but usage of all materials is not required. ", chosen_invalid_materials, chosen_invalid_materials
    elif(choice == "Extra"):
        extra_choices: int = min(_np_rng.poisson(len(template_maker.get_unused_materials(materials_with_invalids))*.2),9) + 1 
        extra_materials =_rng.sample(template_maker.get_unused_materials(materials),extra_choices)
        new_materials = materials_with_invalids + extra_materials
        return f"the stack may be composed of {template_maker.get_str_materials(new_materials)}, but usage of all materials is not required. ", extra_materials + chosen_invalid_materials, chosen_invalid_materials
    elif(choice == "Restrict"):
        unused_materials = template_maker.get_unused_materials(materials)
        restrict_count: int = max(min(_np_rng.poisson(len(unused_materials) * 0.3), len(unused_materials)),1)
        if restrict_count > 0:
            restricted_materials =_rng.sample(unused_materials, restrict_count) + chosen_invalid_materials
            remaining_materials = [mat for mat in unused_materials if mat not in restricted_materials]
            return f"the stack may be composed of any material except {template_maker.get_str_materials(restricted_materials)}. ", remaining_materials, chosen_invalid_materials

def get_additional_errored_constraints(
        materials, 
        thicknesses, 
        extra_materials, 
        num_layers,
        layer_choice,
        num_errors
    ):
    """
    Contradicting additional constraints
            - specific layer less than x but greater than X
            - each layer less than x, specific layer greater than X
            - each layer greater than X, specific layer less than x
            - each layer thicker than x, (req >b layers), whole stack must be less than b*x
            - each layer thinner than x, (req >b layers), whole stack must be greater than b*x
    """

    # Create combined lists for uniform selection
    all_materials_for_constraints = materials + extra_materials
    all_thicknesses_for_constraints = thicknesses + [
        # Generate thickness for each extra material based on random actual material thickness + noise
        max((thicknesses[_rng.randrange(len(thicknesses))] + 
             (_np_rng.poisson(thicknesses[_rng.randrange(len(thicknesses))] * 0.25 / 5) * 5 + 5) *_rng.choice([-1, 1])), 
            5)  # Ensure multiple of 5 and minimum 5
        for _ in extra_materials
    ]
    

    def individual_thickness_contradiction(all_materials_for_constraints, all_thicknesses_for_constraints, thicknesses, num_layers):
        index =_rng.randrange(len(all_materials_for_constraints))
        material = all_materials_for_constraints[index]
        thickness = all_thicknesses_for_constraints[index]
        

        noise_1: int =_np_rng.poisson(thickness * 0.25 / 5) * 5 + 5
        noise_2: int =_np_rng.poisson(thickness * 0.25 / 5) * 5 + 5

        return [
            f"If used, {material} must be thicker than {thickness + noise_1} nm. ",
            f"If used, {material} must be thinner than {thickness - noise_2} nm. "
        ]

    def each_layer_thicker_one_thinner_contradiction(all_materials_for_constraints, all_thicknesses_for_constraints, thicknesses, num_layers):
        options = ["Maximum", "Minimum"]
        weights = [0.5, 0.5]
        choice =_rng.choices(options, weights=weights, k=1)[0]
        noise: int =_np_rng.poisson(max(thicknesses)*.25/5)*5 + 5
        noise_2: int =_np_rng.poisson(max(thicknesses)*.25/5)*5 + 5

        index =_rng.randrange(len(all_materials_for_constraints))
        material = all_materials_for_constraints[index]
        
        if(choice == "Maximum"):
            maximum = max(thicknesses) + noise
            maxiermum = maximum + noise_2
            return [f"Each of the layers in the structure must be thinner than {maximum} nm. ", f"If used, {material} must be thicker than {maxiermum} nm. "]
        elif(choice == "Minimum"):
            minimum = max(min(thicknesses) - noise_2,10)
            miniermum = max(minimum - noise_2,5)
            return [f"Each of the layers in the structure must be thicker than {minimum} nm. ", f"If used, {material} must be thinner than {miniermum} nm. "]
        
    def total_thickness_thicker_each_thinner_contradiction(all_materials_for_constraints, all_thicknesses_for_constraints, thicknesses, num_layers):
        noise: int =_np_rng.poisson(sum(thicknesses)*.25/5)*5 + 5
        noise_2: int =_np_rng.poisson(sum(thicknesses)*.25/5)*5 + 10
        options = ["Maximum", "Minimum"]
        weights = [0.5, 0.5]
        choice =_rng.choices(options, weights=weights, k=1)[0]

        if(choice == "Maximum"):
            stack_maximum = sum(thicknesses) + noise
            each_layer_minimum = max(math.ceil((stack_maximum/num_layers + noise_2)/5)*5,10)
            return [f"The total thickness of the structure must be less than {stack_maximum} nm. ", f"Each of the layers in the structure must be thicker than {each_layer_minimum} nm. "]
        elif(choice == "Minimum"):
            stack_minimum = max(sum(thicknesses) - noise,20)
            each_layer_maximum = math.floor(max(stack_minimum/num_layers - noise_2,5)/5)*5
            return [f"The total thickness of the structure must be greater than {stack_minimum} nm. ", f"Each of the layers in the structure must be thinner than {each_layer_maximum} nm. "]
    
    # collect outputs here (if you already have this, keep yours)
    invalid_pieces = []

    # functions to choose from (no repetitions)
    _subfns = [
        individual_thickness_contradiction,
        each_layer_thicker_one_thinner_contradiction,
        total_thickness_thicker_each_thinner_contradiction,
    ]

    if layer_choice != "Exact":
        _subfns = [
            fn for fn in _subfns
            if fn is not total_thickness_thicker_each_thinner_contradiction
        ]
    # choose uniformly without replacement, up to num_errors (will be <= 3)
    _to_call = _rng.sample(_subfns, k=min(num_errors, len(_subfns)))
    
    

    # call each chosen function once
    for fn in _to_call:
        invalid_pieces.extend(
            fn(all_materials_for_constraints,
            all_thicknesses_for_constraints,
            thicknesses,
            num_layers)
        )

    return invalid_pieces
        
        


def get_template_with_errors(num_layers, materials, thicknesses, incidence_angle, color_rgb, picked_seed = 42, verbose = False):

    _rng.seed(picked_seed)
    global _np_rng
    _np_rng = np.random.default_rng(picked_seed)
    num_errors = min(max(1,_np_rng.poisson(0.2)),4)

    procedural_template, layer_choice = template_maker.get_layers_fragment(num_layers)
    chosen_invalid_materials = []

    if(_rng.random() <= 0.4):
        procedural_materials, extra_materials, chosen_invalid_materials= get_errored_materials_fragment(materials)
        procedural_template += procedural_materials
        num_errors -= 1
    else:
        procedural_materials, extra_materials = template_maker.get_materials_fragment(materials)
        procedural_template += procedural_materials
        if(num_errors >= 3):
            num_errors = 3

    
    #procedural_template += f"At {incidence_angle} degrees, "
    procedural_template += template_maker.get_color_identity(color_rgb)
    additional_constraints_list = template_maker.additional_constraints(
        materials, 
        thicknesses, 
        extra_materials, 
        avg_adj_constraints = 0.025*.5, #average adjacent constraint count
        avg_rel_tkns_constraints = 0.025*.5, #average relative thickness constraint count
        avg_indv_tkns_constraints =0.8*.5, #average indivudal thickness constraint count
        ttl_tkns_constraint_prob = 0.55*.5, #total thickness constraint probability
        lyr_id_constaint_prob = 0.02*.5, #layer id constraint probability
        force_zero = False
    )
    invalid_constraints = get_additional_errored_constraints(
        materials, 
        thicknesses, 
        extra_materials, 
        num_layers,
        layer_choice,
        num_errors
    )
    additional_constraints_list += invalid_constraints
    _random.shuffle(additional_constraints_list)
    procedural_template += "".join(additional_constraints_list)
    

    if verbose:
        print(f"extra materials: {extra_materials}")
        print(f"invalid additional constraints: {invalid_constraints}")
        print(f"chosen invalid materials: {chosen_invalid_materials}")
        print(f"exact layer flag: {(layer_choice == 'Exact')}")
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
    for sed in range(1100):
        prompt_seed = sed
        print(get_template_with_errors(num_layers, materials, thicknesses, incidence_angle, color_rgb, picked_seed=prompt_seed, verbose=True))
        print("=" * 60)
