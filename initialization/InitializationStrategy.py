import torch
from abc import ABC, abstractmethod
from typing import Tuple

class InitializationStrategy(ABC):
    """Abstract base class for superellipse initialization strategies"""
    @abstractmethod
    def initialize(self, num_ellipses: int, image_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize parameters for multiple superellipses
        
        Args:
            num_ellipses: Number of superellipses to initialize
            image_size: Size of the image (assumed square)
            square_size: Size of the target square
            
        Returns:
            Tuple of (centers_x, centers_y, radii, ns, rotations)
        """
        pass