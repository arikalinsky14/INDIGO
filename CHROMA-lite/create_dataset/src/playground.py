import os
import numpy as np
import random as _random
import matplotlib.pyplot as plt
from color_palette_manager import PaletteMatcher as PM

# --- Setup random generators ---
_rng = _random.Random()
_np_rng = np.random.default_rng()

# --- Register and activate your palette ---
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

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")

def show_color(rgb, common_name, full_name, hex_code):
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(np.ones((10, 10, 3)) * np.array(rgb) / 255)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{common_name}\n{hex_code}", fontsize=10)
    plt.show(block=False)
    print("\nPress Enter to continue to next sample...")
    input()
    clear_console()
    plt.close(fig)

# --- Generate and display random colors ---
num_samples = 100  # adjust as desired
for i in range(num_samples):
    color_rgb = [int(_rng.randint(0, 255)) for _ in range(3)]
    r, g, b = color_rgb
    hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    print(f"\nSample {i+1}")
    print(f"RGB: ({r}, {g}, {b})")

    common_name = PM.find(r, g, b)
    with PM.use("__full__"):
        full_name = PM.find(r, g, b)

    print(f"Closest common name: {common_name}")
    print(f"Closest full name:   {full_name}")
    print(f"HEX code:            {hex_code}")

    show_color(color_rgb, common_name, full_name, hex_code)

    
