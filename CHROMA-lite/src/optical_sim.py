"""
Optical Simulation Module for CHROMA-Lite

Provides thin-film optical simulation using jaxlayerlumos to compute
reflectance spectra and convert them to sRGB colors.

This module is used during evaluation to compute the predicted color
from a generated thin-film structure, enabling CIEDE2000 comparison
against ground truth colors.

Requirements:
    pip install jaxlayerlumos jax scipy

Usage:
    from src.optical_sim import OpticalSimulator
    
    sim = OpticalSimulator()
    pred_sRGB = sim.compute_color(['SiO2', 'Ag', 'TiO2'], [100, 50, 75])
"""

import math
from typing import List, Tuple, Optional
import numpy as np

# JAX imports for optical simulation
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import scipy.constants as scic
    from jaxlayerlumos.utils_materials import get_n_k
    from jaxlayerlumos.jaxlayerlumos import stackrt_n_k
    import jaxlayerlumos.colors.composite as jll_colors_composite
    JAX_AVAILABLE = True
except ImportError as e:
    JAX_AVAILABLE = False
    _IMPORT_ERROR = str(e)


def spectrum_to_sRGB(wavelengths_nm: np.ndarray, spectrum: np.ndarray) -> List[int]:
    """
    Convert reflectance spectrum to sRGB [R, G, B] in 0-255 range.
    
    Args:
        wavelengths_nm: Wavelength array in nanometers
        spectrum: Reflectance spectrum (0-1 range)
    
    Returns:
        List of [R, G, B] integers in 0-255 range
    """
    if not JAX_AVAILABLE:
        raise RuntimeError(f"jaxlayerlumos not available: {_IMPORT_ERROR}")
    
    assert wavelengths_nm.shape[0] == spectrum.shape[0]
    
    # Filter to valid range for color matching functions
    indices = np.logical_and(wavelengths_nm > 360, wavelengths_nm < 830)
    
    sRGB = jll_colors_composite.spectrum_to_sRGB(
        jnp.array(wavelengths_nm[indices]),
        jnp.array(spectrum[indices]),
        use_clipping=True
    )
    
    sRGB = np.array(sRGB)
    sRGB *= 255
    sRGB = sRGB.flatten()[:3]
    sRGB = np.round(sRGB).astype(int).tolist()
    
    return sRGB


class OpticalSimulator:
    """
    Thin-film optical simulation for structural coloration.
    
    Computes reflectance spectra using transfer matrix method and converts
    to sRGB colors. Matches the parameters used in dataset generation.
    
    Attributes:
        incidence_angle: Angle of incidence in degrees (default: 0 = normal)
        num_points: Number of wavelength points in spectrum
        wavelength_nm: Wavelength array in nanometers
    """
    
    def __init__(
        self,
        incidence_angle: int = 0,
        wavelength_range: Tuple[int, int] = (300, 900),
        num_points: int = 128
    ):
        """
        Initialize the optical simulator.
        
        Args:
            incidence_angle: Angle of incidence in degrees
            wavelength_range: (min_wavelength, max_wavelength) in nm
            num_points: Number of spectral points
        """
        if not JAX_AVAILABLE:
            raise RuntimeError(f"jaxlayerlumos not available: {_IMPORT_ERROR}")
        
        self.incidence_angle = incidence_angle
        self.num_points = num_points
        
        # Compute frequency/wavelength vectors
        start_freq = scic.c / (wavelength_range[1] * scic.nano)
        stop_freq = scic.c / (wavelength_range[0] * scic.nano)
        
        self.freq_vector = jnp.linspace(start_freq, stop_freq, num_points)
        self.wavelength_vector = scic.c / self.freq_vector
        self.wavelength_nm = np.array(self.wavelength_vector) / scic.nano
    
    def compute_reflectance(
        self,
        materials: List[str],
        thicknesses: List[int]
    ) -> np.ndarray:
        """
        Compute reflectance spectrum for a thin-film stack.
        
        Args:
            materials: List of material names (e.g., ['SiO2', 'Ag', 'TiO2'])
            thicknesses: List of thicknesses in nm (e.g., [100, 50, 75])
        
        Returns:
            R_avg: Average reflectance spectrum (TE+TM)/2, shape (num_points,)
        """
        if len(materials) == 0:
            # Empty structure - return zero reflectance
            return np.zeros(self.num_points)
        
        # Add Air superstrate and FusedSilica substrate
        full_materials = ['Air'] + list(materials) + ['FusedSilica']
        full_thicknesses = [0] + list(thicknesses) + [0]
        
        # Handle material name aliases
        full_materials = ['AZO-Zarei' if m == 'AZO' else m for m in full_materials]
        
        # Get refractive indices
        n_matrix = get_n_k(full_materials, self.freq_vector)
        
        # Convert thicknesses to meters
        d_jax = jnp.array(full_thicknesses, dtype=jnp.float32) * scic.nano
        thetas = jnp.array([self.incidence_angle], dtype=jnp.float32)
        
        # Compute transfer matrix
        R_TE, T_TE, R_TM, T_TM = stackrt_n_k(n_matrix, d_jax, self.freq_vector, thetas)
        
        # Average polarizations
        R_avg = (R_TE[0] + R_TM[0]) / 2
        
        return np.array(R_avg)
    
    def compute_color(
        self,
        materials: List[str],
        thicknesses: List[int]
    ) -> List[int]:
        """
        Compute sRGB color for a thin-film stack.
        
        Args:
            materials: List of material names
            thicknesses: List of thicknesses in nm
        
        Returns:
            List of [R, G, B] integers in 0-255 range
        """
        R = self.compute_reflectance(materials, thicknesses)
        return spectrum_to_sRGB(self.wavelength_nm, R)


def is_available() -> bool:
    """Check if optical simulation is available (jaxlayerlumos installed)."""
    return JAX_AVAILABLE


def get_import_error() -> Optional[str]:
    """Get the import error message if jaxlayerlumos is not available."""
    if JAX_AVAILABLE:
        return None
    return _IMPORT_ERROR
