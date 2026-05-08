import numpy as np

import templates


char_double_backslash = '\\\\'


def get_str_materials_thicknesses(materials, thicknesses):
    assert len(materials) == len(thicknesses)

    str_materials_thicknesses = ', '.join([f'({material}, {thickness:d} nm)' for material, thickness in zip(materials, thicknesses)])
    str_materials_thicknesses = '[' + str_materials_thicknesses + ']'
    return str_materials_thicknesses


def get_str_rgb(sRGB):
    assert len(sRGB) == 3

    str_rgb = ', '.join([f'{elem:d}' for elem in sRGB])
    str_rgb = '[' + str_rgb + ']'
    return str_rgb


def get_instruction(index_template):
    examples = [
        [
            4,
            ['GaAs', 'TiN', 'Pd', 'InP'],
            [40, 105, 195, 95],
            0,
            [211, 219, 181],
            'a pale olive hue',
        ],
        [
            6,
            ['TiO2', 'Au', 'GaInP', 'W', 'Al2O3', 'aSi'],
            [10, 120, 150, 190, 10, 105],
            30,
            [248, 200, 89],
            'a golden yellow color',
        ],
        [
            8,
            ['ZnO', 'GaInP', 'TiO2', 'Pt', 'Mn', 'SiO2', 'Pd', 'TiO2'],
            [120, 130, 125, 145, 75, 5, 125, 35],
            60,
            [61, 137, 140],
            'a teal color',
        ],
    ]

    list_str_examples = []
    for ind_example, example in enumerate(examples):
        num_layers, materials, thicknesses, incidence_angle, rgb_value, color_name = example

        str_structure_information = get_prompt_structure_information(materials, thicknesses, incidence_angle, rgb_value)
        str_template = templates.get_template(index_template, num_layers, materials, thicknesses, incidence_angle, color_name)

        str_example = f'Example {ind_example + 1}\n\nInput:\n{str_structure_information}\n\nOutput:\n{str_template}'
        list_str_examples.append(str_example)

    str_examples = '\n\n'.join(list_str_examples)

    return f"""Generate a single-paragraph plain-text description of a multi-layer thin-film optical structure using the given number of layers, material layout, and reflected color at a specified incidence angle. The description must mention the total number of layers and express the reflected color using a standard color name. Exclude any numeric RGB values or specific layer thicknesses. To support generalization for inverse design algorithm development, randomly shuffle the order of the materials provided in the input.
    
The following examples provide the optical structure inputs and their corresponding desired descriptive outputs.

{str_examples}

While generating the output, you can modify the wording as long as the content stays consistent."""


def get_prompt_structure_information(materials, thicknesses, incidence_angle, rgb_value):
    assert len(materials) == len(thicknesses)
    assert len(rgb_value) == 3

    num_layers = len(materials)

    str_materials_thicknesses = get_str_materials_thicknesses(materials, thicknesses)
    str_rgb = get_str_rgb(rgb_value)

    return f"""Number of layers: {num_layers}
Material layout and layer thicknesses: {str_materials_thicknesses}
Incidence angle: {incidence_angle} degrees
RGB value: {str_rgb}"""


if __name__ == '__main__':
    print(get_instruction(1))
    print('')
    print(get_instruction(2))
    print('')
    print(get_instruction(3))
    print('')
    print(get_instruction(4))
    print('')
    print(get_instruction(5))
    print('')
    print(get_instruction(6))
    print('')
    print(get_instruction(7))
    print('')
    print(get_instruction(8))
    print('')

    materials = ['TiO2', 'Ag', 'TiO2']
    thicknesses = [10, 20, 30]
    incidence_angle = 45
    rgb_value = [100, 30, 150]

    print(get_prompt_structure_information(materials, thicknesses, incidence_angle, rgb_value))
    print('')
