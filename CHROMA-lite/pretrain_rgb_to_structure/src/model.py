"""
Model Architecture - Simple Feedforward MLP

Input: RGB [B, 3] + structure matrix [B, 25, 8] (flattened to 200)
Output: Logits over 1001 tokens (1000 material-thickness pairs + EOS)

Architecture:
    Input (203) -> Dense(d_model) -> ReLU -> Dropout
                -> [Dense(d_model) -> ReLU -> Dropout] x (n_layers - 1)
                -> Dense(vocab_size)

The structure matrix implicitly encodes which step we're at:
  - At step 0: all zeros -> predict layer 0
  - At step k: layers 0..k-1 filled -> predict layer k
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# FIXED: Use absolute imports from top-level src/ module
from src.materials_vocab import (
    NUM_MATERIALS, MAX_LAYERS, VOCAB_SIZE,
    EOS_TOKEN, decode_token, MATERIAL_TO_IDX, normalize_thickness
)


def _find_repo_root() -> Path:
    """Find chroma-lite repo root."""
    current = Path(__file__).resolve().parent
    while current.parent != current:
        if (current / "src").exists() and (current / "create_dataset").exists():
            return current
        current = current.parent
    return Path.cwd()


class ModelConfig:
    """
    Configuration for the feedforward MLP model.
    
    Hyperparameters:
        d_model: Hidden layer dimension
        n_layers: Number of hidden layers in the MLP
        dropout: Dropout rate
    """
    def __init__(
        self, 
        d_model: int = 256, 
        n_layers: int = 4, 
        dropout: float = 0.1,
        learning_rate: float = 1e-4, 
        batch_size: int = 32, 
        limit_examples: Optional[int] = None,
        epochs: int = 1,
        # Legacy params (ignored, for checkpoint compatibility)
        n_heads: int = None,
        ff_dim: int = None,
        max_layers: int = None,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.vocab_size = VOCAB_SIZE
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.limit_examples = limit_examples
        self.epochs = epochs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'limit_examples': self.limit_examples,
            'epochs': self.epochs,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ModelConfig':
        # Filter out legacy params that might be in old checkpoints
        valid_keys = {'d_model', 'n_layers', 'dropout', 'learning_rate', 
                      'batch_size', 'limit_examples', 'epochs'}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def tag(self) -> str:
        """Generate hyperparameter tag for checkpoint directory naming."""
        base = f"mlp_d{self.d_model}_L{self.n_layers}_do{self.dropout}_lr{self.learning_rate}_bs{self.batch_size}_ep{self.epochs}"
        if self.limit_examples is not None:
            base += f"_lim{self.limit_examples}"
        return base
    
    def default_checkpoint_dir(self) -> Path:
        """Return default checkpoint directory based on hyperparameters."""
        repo_root = _find_repo_root()
        return repo_root / "pretrain_rgb_to_structure" / "data" / "checkpoints" / self.tag()


class ThinFilmMLP(nn.Module):
    """
    Simple feedforward MLP for thin-film structure prediction.
    
    Takes RGB color + current structure state, outputs logits for next layer token.
    
    Input dimensions:
        - RGB: 3 values (normalized 0-1)
        - Structure matrix: 25 * 8 = 200 values (flattened)
        - Total input: 203 values
    
    Output: vocab_size logits (1001)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input: RGB (3) + flattened structure (25 * 8 = 200) = 203
        input_dim = 3 + NUM_MATERIALS * MAX_LAYERS
        
        # Build MLP layers
        layers = []
        
        # First layer: input -> d_model
        layers.append(nn.Linear(input_dim, config.d_model))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))
        
        # Hidden layers: d_model -> d_model
        for _ in range(config.n_layers - 1):
            layers.append(nn.Linear(config.d_model, config.d_model))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
        
        # Output layer: d_model -> vocab_size
        layers.append(nn.Linear(config.d_model, config.vocab_size))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, rgb: torch.Tensor, structure_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            rgb: [B, 3] normalized RGB values
            structure_matrix: [B, 25, 8] current structure state
        
        Returns:
            logits: [B, vocab_size]
        """
        batch_size = rgb.size(0)
        
        # Flatten structure matrix: [B, 25, 8] -> [B, 200]
        structure_flat = structure_matrix.view(batch_size, -1)
        
        # Concatenate inputs: [B, 3] + [B, 200] -> [B, 203]
        x = torch.cat([rgb, structure_flat], dim=1)
        
        # Forward through MLP
        logits = self.network(x)
        
        return logits  # [B, vocab_size]


# Alias for backward compatibility
ThinFilmTransformer = ThinFilmMLP


def compute_loss(
    model: ThinFilmMLP, 
    rgb: torch.Tensor, 
    structure_matrix: torch.Tensor, 
    target_token: torch.Tensor,
    step: int = None  # Ignored - kept for API compatibility
) -> Dict[str, torch.Tensor]:
    """
    Compute loss for a prediction.
    
    Args:
        model: The MLP model
        rgb: [B, 3] RGB values
        structure_matrix: [B, 25, 8] current structure
        target_token: [B] target token IDs
        step: Ignored (kept for API compatibility with training script)
    
    Returns:
        dict with 'loss' and 'accuracy'
    """
    logits = model(rgb, structure_matrix)  # [B, vocab_size]
    
    loss = F.cross_entropy(logits, target_token)
    accuracy = (logits.argmax(dim=-1) == target_token).float().mean()
    
    return {'loss': loss, 'accuracy': accuracy}


def generate_structure(
    model: ThinFilmMLP, 
    rgb: torch.Tensor, 
    device: torch.device, 
    max_layers: int = MAX_LAYERS,
    sample: bool = False,
    temperature: float = 1.0,
    generator: torch.Generator = None,
) -> Tuple[List[str], List[int], str]:
    """
    Autoregressively generate a thin-film structure for a target RGB color.
    
    Args:
        model: Trained MLP model
        rgb: [3] normalized RGB tensor
        device: torch device
        max_layers: Maximum number of layers to generate
        sample: If True, use stochastic sampling instead of greedy argmax
        temperature: Sampling temperature (higher = more random). Only used
                     when sample=True. Values < 1 sharpen, > 1 flatten.
        generator: Optional torch.Generator for reproducible sampling
    
    Returns:
        Tuple of (materials, thicknesses, stop_reason)
        stop_reason is 'EOS' if model predicted end, 'MAX_LEN' if hit limit
    """
    model.eval()
    
    # Initialize empty structure
    structure = torch.zeros(NUM_MATERIALS, MAX_LAYERS, device=device)
    materials, thicknesses = [], []
    rgb = rgb.to(device)
    
    with torch.no_grad():
        for step in range(max_layers):
            # Get prediction
            logits = model(rgb.unsqueeze(0), structure.unsqueeze(0))
            
            if sample and temperature > 0:
                # Stochastic sampling with temperature
                probs = F.softmax(logits[0] / temperature, dim=-1)
                token_id = torch.multinomial(probs, 1, generator=generator).item()
            else:
                # Greedy argmax
                token_id = logits.argmax(dim=-1).item()
            
            # Check for EOS
            if token_id == EOS_TOKEN:
                return materials, thicknesses, 'EOS'
            
            # Decode token and add to structure
            material, thickness, _ = decode_token(token_id)
            materials.append(material)
            thicknesses.append(thickness)
            
            # Update structure matrix for next iteration
            structure[MATERIAL_TO_IDX[material], step] = normalize_thickness(thickness)
    
    return materials, thicknesses, 'MAX_LEN'