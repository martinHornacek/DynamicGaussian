from dataclasses import dataclass
from initialization.CircularStrategy import CircularStrategy
from initialization.InitializationStrategy import InitializationStrategy

@dataclass
class ModelConfig:
    """Configuration class for model parameters and ablation studies"""
    num_ellipses: int = 4
    enable_rotation: bool = False
    enable_n_parameter: bool = True
    fixed_n_value: float = 4.0
    initialization_strategy: InitializationStrategy = CircularStrategy()
    
    # Penalty weights
    w_overlap: float = 0.5
    w_n_regularization: float = 0.005
    w_radius: float = 0.5
    
    # Training parameters
    learning_rate: float = 0.01
    num_epochs: int = 10000
    frame_interval: int = 50