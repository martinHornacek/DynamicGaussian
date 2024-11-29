import numpy as np
import torch
from typing import Tuple
from initialization.InitializationStrategy import InitializationStrategy

class UniformStrategy(InitializationStrategy):
    """
    Initialize superellipses uniformly across the image
    """
    def __init__(
        self, 
        initial_n: float = 3.0, 
        initial_radius_factor: float = 0.25
    ):
        self.initial_n = initial_n
        self.initial_radius_factor = initial_radius_factor
    
    def initialize(
        self, 
        num_ellipses: int, 
        image_size: int, 
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly distribute superellipses across the image
        """
        # Random x and y coordinates within image bounds
        centers_x = torch.rand(num_ellipses) * image_size
        centers_y = torch.rand(num_ellipses) * image_size
        
        # Radius size as a fraction of image size
        radii = torch.ones(num_ellipses) * (image_size * self.initial_radius_factor)
        
        ns = torch.ones(num_ellipses) * self.initial_n
        rotations = torch.zeros(num_ellipses)
        
        return centers_x, centers_y, radii, ns, rotations