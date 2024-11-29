import numpy as np
import torch
from typing import Tuple, Optional, Callable
from initialization.InitializationStrategy import InitializationStrategy

class RandomStrategy(InitializationStrategy):
    """
    Initialize superellipses with random positions considering shape constraints
    """
    def __init__(
        self, 
        margin: float = 0.1,  # Margin as fraction of image size
        initial_n: float = 3.0,
        radius_range: Tuple[float, float] = (0.05, 0.2),  # Radius as fraction of image size
        placement_constraint: Optional[Callable[[np.ndarray, float, float], bool]] = None
    ):
        """
        Args:
            margin (float): Margin around image edges
            initial_n (float): Initial superellipse shape parameter
            radius_range (Tuple[float, float]): Radius range as fraction of image size
            placement_constraint (Callable, optional): Custom function to validate ellipse placement
        """
        self.margin = margin
        self.initial_n = initial_n
        self.radius_range = radius_range
        self.placement_constraint = placement_constraint or self._default_placement_constraint
    
    def _default_placement_constraint(
        self, 
        ground_truth: np.ndarray, 
        x: float, 
        y: float
    ) -> bool:
        """
        Default placement constraint: ensure ellipse is mostly within shape
        
        Args:
            ground_truth (np.ndarray): Binary ground truth image
            x (float): X-coordinate of ellipse center
            y (float): Y-coordinate of ellipse center
        
        Returns:
            bool: Whether placement is valid
        """
        # Ensure point is within the shape's non-zero region
        height, width = ground_truth.shape
        x, y = int(x), int(y)
        
        # Check if point is within image bounds
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        
        # Check if point is within the shape (close to zero/black region)
        return ground_truth[y, x] == 0
    
    def initialize(
        self, 
        num_ellipses: int, 
        image_size: int, 
        ground_truth: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize superellipses with random positioning
        
        Args:
            num_ellipses (int): Number of superellipses to place
            image_size (int): Total image size
            square_size (int, optional): Reference size (not used in this version)
            ground_truth (np.ndarray, optional): Ground truth image for placement constraints
        
        Returns:
            Tuple of tensor parameters for superellipses
        """
        # Validate inputs
        if ground_truth is None:
            raise ValueError("Ground truth image is required for constrained random initialization")
        
        margin_pixels = int(image_size * self.margin)
        
        # Prepare lists to store valid superellipse parameters
        centers_x = []
        centers_y = []
        radii = []
        
        # Maximum attempts to place ellipses
        max_attempts = num_ellipses * 100
        attempts = 0
        
        while len(centers_x) < num_ellipses and attempts < max_attempts:
            # Generate random coordinates
            x = torch.rand(1).item() * (image_size - 2 * margin_pixels) + margin_pixels
            y = torch.rand(1).item() * (image_size - 2 * margin_pixels) + margin_pixels
            
            # Check placement constraint
            if self.placement_constraint(ground_truth, x, y):
                centers_x.append(x)
                centers_y.append(y)
                
                # Radius based on image size
                min_radius = image_size * self.radius_range[0]
                max_radius = image_size * self.radius_range[1]
                radii.append(torch.rand(1).item() * (max_radius - min_radius) + min_radius)
            
            attempts += 1
        
        # Ensure we have enough ellipses
        if len(centers_x) < num_ellipses:
            raise ValueError(f"Could only place {len(centers_x)} ellipses out of {num_ellipses} requested")
        
        # Convert to tensors
        centers_x = torch.tensor(centers_x)
        centers_y = torch.tensor(centers_y)
        radii = torch.tensor(radii)
        
        ns = torch.ones(num_ellipses) * self.initial_n
        rotations = torch.rand(num_ellipses) * 2 * np.pi - np.pi
        
        return centers_x, centers_y, radii, ns, rotations