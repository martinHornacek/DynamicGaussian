import numpy as np
import torch
from typing import Tuple
from initialization.InitializationStrategy import InitializationStrategy

class CircularStrategy(InitializationStrategy):
    """
    Initialize superellipses in a circular pattern around the center
    with configuration independent of specific ground truth shape
    """
    def __init__(
        self, 
        radius_factor: float = 0.75,  # Fraction of image size
        initial_n: float = 3.0, 
        initial_radius_factor: float = 0.25,  # Fraction of image size
        spread_factor: float = 0.9  # Controls how much of the available radius to use
    ):
        """
        Args:
            radius_factor (float): Fraction of image size for placement radius
            initial_n (float): Initial exponent for superellipse shape
            initial_radius_factor (float): Initial radius as fraction of image size
            spread_factor (float): Controls spacing of superellipses
        """
        self.radius_factor = radius_factor
        self.initial_n = initial_n
        self.initial_radius_factor = initial_radius_factor
        self.spread_factor = spread_factor
    
    def initialize(
        self, 
        num_ellipses: int, 
        image_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize superellipse positions and parameters
        
        Args:
            num_ellipses (int): Number of superellipses to place
            image_size (int): Total size of the image
            square_size (int, optional): Size of reference shape (not used in this version)
        
        Returns:
            Tuple of tensor parameters for superellipses
        """
        # Calculate placement parameters based on image size
        angles = torch.linspace(0, 2*np.pi, num_ellipses+1)[:-1]
        
        # Radius is now a fraction of image size, not dependent on square_size
        placement_radius = image_size * self.radius_factor * self.spread_factor / 2
        center = image_size // 2
        
        # Calculate center positions in a circular arrangement
        centers_x = center + placement_radius * torch.cos(angles)
        centers_y = center + placement_radius * torch.sin(angles)
        
        # Radius size as a fraction of image size
        radii = torch.ones(num_ellipses) * (image_size * self.initial_radius_factor)
        
        # Other parameters remain consistent
        ns = torch.ones(num_ellipses) * self.initial_n
        rotations = torch.zeros(num_ellipses)
        
        return centers_x, centers_y, radii, ns, rotations