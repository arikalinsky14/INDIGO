"""
CHROMA-Lite Model Package - pretrain_rgb_to_structure

This subpackage contains the MLP model for RGB -> structure prediction.
Shared modules (materials_vocab, dataset, optical_sim) are in the top-level src/.
"""

from .model import ModelConfig, ThinFilmMLP, ThinFilmTransformer, compute_loss, generate_structure

__all__ = [
    'ModelConfig',
    'ThinFilmMLP', 
    'ThinFilmTransformer',
    'compute_loss',
    'generate_structure',
]