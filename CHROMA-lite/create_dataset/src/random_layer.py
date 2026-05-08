import pandas as pd
import numpy as onp
import os
import scipy.constants as scic
import jax.numpy as jnp
import jax

from jaxlayerlumos.utils_materials import get_n_k
from jaxlayerlumos.jaxlayerlumos import stackrt_n_k


jax.config.update("jax_enable_x64", True)


class RandomLayerSimulation:
    def __init__(self, num_layers=4, incidence_angle=0, wavelength_range=(300, 900), num_points=128, output_dir="../data", seed=42):
        self.num_layers = num_layers
        self.incidence_angle = incidence_angle
        self.num_points = num_points

        self.output_dir = output_dir
        self.output_dir = os.path.join(
            self.output_dir, f'layers_{self.num_layers:02d}_angle_{self.incidence_angle:02d}')

        os.makedirs(self.output_dir, exist_ok=True)

        start_freq = scic.c / (wavelength_range[1] * scic.nano)
        stop_freq = scic.c / (wavelength_range[0] * scic.nano)

        self.freq_vector = jnp.linspace(start_freq, stop_freq, num_points)
        self.wavelength_vector = scic.c / self.freq_vector

        self.materials = [
            'Ag',
            'Al',
            'Al2O3',
            'Au',
            'AZO-Zarei',
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
            'ZnO',
        ]
        self.thickness_range = onp.arange(5, 201, 5)

        self.seed = seed
        self.random_state = onp.random.RandomState(seed)

    def random_materials_and_thicknesses(self):
        layer_materials = list(self.random_state.choice(self.materials, size=self.num_layers, replace=False))

        layer_thicknesses = self.random_state.choice(self.thickness_range, size=self.num_layers, replace=True)
        layer_thicknesses = [thickness for thickness in layer_thicknesses]

        return layer_materials, layer_thicknesses

    def create_n_matrix(self, layer_materials):
        return get_n_k(layer_materials, self.freq_vector)

    def calculate_tr(self, n_matrix, layer_thicknesses):
        d_jax = jnp.array(layer_thicknesses, dtype=jnp.float32) * scic.nano
        thetas = jnp.array([self.incidence_angle], dtype=jnp.float32)

        R_TE, T_TE, R_TM, T_TM = stackrt_n_k(n_matrix, d_jax, self.freq_vector, thetas)

        R_avg = (R_TE[0] + R_TM[0]) / 2
        T_avg = (T_TE[0] + T_TM[0]) / 2
        return R_avg, T_avg

    def run_simulation(self, iterations=10000):
        data = []
        wavelength_vector_np = onp.array(self.wavelength_vector)

        for ind_iteration in range(0, iterations):
            layer_materials, layer_thicknesses = self.random_materials_and_thicknesses()
            n_matrix = self.create_n_matrix(['Air'] + layer_materials + ['FusedSilica'])
            R_avg, T_avg = self.calculate_tr(n_matrix, [0] + layer_thicknesses + [0])

            R_avg_np = onp.array(R_avg)
            T_avg_np = onp.array(T_avg)
            assert R_avg_np.ndim == T_avg_np.ndim == 1
            assert R_avg_np.shape[0] == T_avg_np.shape[0] == wavelength_vector_np.shape[0]

            material_names = "_".join(layer_materials)
            thicknesses = "_".join([f"{thickness}nm" for thickness in layer_thicknesses])

            for i, wavelength in enumerate(wavelength_vector_np):
                data.append([ind_iteration, material_names, thicknesses, wavelength, R_avg_np[i], T_avg_np[i]])

        df = pd.DataFrame(data, columns=["Index", "Materials", "Thicknesses", "Wavelength (m)", "R", "T"])

        filename = os.path.join(
            self.output_dir,
            f"TR_simulations_layers_{self.num_layers:02d}_angle_{self.incidence_angle:02d}_seed_{self.seed:05d}.parquet"
        )
        df.to_parquet(filename, index=False, compression="gzip")
