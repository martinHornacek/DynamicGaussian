import numpy as np
import torch
from typing import Tuple, Optional, Callable
from initialization.InitializationStrategy import InitializationStrategy

class PerimeterStrategy(InitializationStrategy):
    """
    Initialize superellipses along the perimeter of a shape
    Supports custom shape boundary calculation
    """
    def __init__(
        self, 
        edge_padding: float = 0.05,  # Padding as fraction of image size
        initial_n: float = 3.0,
        boundary_finder: Optional[Callable[[np.ndarray], dict]] = None
    ):
        """
        Args:
            edge_padding (float): Padding distance from shape boundary
            initial_n (float): Initial superellipse shape parameter
            boundary_finder (Callable, optional): Custom function to find shape boundaries
        """
        self.edge_padding = edge_padding
        self.initial_n = initial_n
        self.boundary_finder = boundary_finder or self._default_boundary_finder
    
    def _default_boundary_finder(self, ground_truth: np.ndarray) -> dict:
        """
        Default method to find shape boundaries in a binary image
        
        Args:
            ground_truth (np.ndarray): Binary ground truth image
        
        Returns:
            dict with shape boundary information
        """
        # Find where the shape transitions from 0 to 1 or 1 to 0
        shape_mask = ground_truth == 0
        rows, cols = np.where(shape_mask)
        
        return {
            'left': np.min(cols),
            'right': np.max(cols),
            'top': np.min(rows),
            'bottom': np.max(rows),
            'center_x': (np.min(cols) + np.max(cols)) / 2,
            'center_y': (np.min(rows) + np.max(rows)) / 2,
            'width': np.max(cols) - np.min(cols),
            'height': np.max(rows) - np.min(rows)
        }
    
    def initialize(
        self, 
        num_ellipses: int, 
        image_size: int, 
        ground_truth: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize superellipses along shape perimeter
        
        Args:
            num_ellipses (int): Number of superellipses to place
            image_size (int): Total image size
            ground_truth (np.ndarray, optional): Ground truth image for boundary detection
        
        Returns:
            Tuple of tensor parameters for superellipses
        """
        # Validate inputs
        if ground_truth is None:
            raise ValueError("Ground truth image is required for perimeter initialization")
        
        # Find shape boundaries
        bounds = self.boundary_finder(ground_truth)
        
        # Calculate padding (as fraction of image size)
        padding = image_size * self.edge_padding
        
        # Prepare lists to store superellipse parameters
        centers_x = []
        centers_y = []
        rotations = []
        
        # Calculate total perimeter and spacing
        perimeter_width = bounds['right'] - bounds['left']
        perimeter_height = bounds['bottom'] - bounds['top']
        total_perimeter = 2 * (perimeter_width + perimeter_height)
        spacing = total_perimeter / num_ellipses
        
        placed_ellipses = 0
        current_pos = 0
        
        while placed_ellipses < num_ellipses:
            # Calculate position along perimeter
            perimeter_pos = current_pos * spacing
            
            # Determine which edge we're on and calculate center coordinates
            if perimeter_pos < perimeter_width:  # Top edge
                x = bounds['left'] + perimeter_pos
                y = bounds['top'] - padding
                rot = 0  # Horizontal orientation
            elif perimeter_pos < perimeter_width + perimeter_height:  # Right edge
                x = bounds['right'] + padding
                y = bounds['top'] + (perimeter_pos - perimeter_width)
                rot = np.pi / 2  # Vertical orientation
            elif perimeter_pos < 2 * perimeter_width + perimeter_height:  # Bottom edge
                x = bounds['right'] - (perimeter_pos - (perimeter_width + perimeter_height))
                y = bounds['bottom'] + padding
                rot = 0  # Horizontal orientation
            else:  # Left edge
                x = bounds['left'] - padding
                y = bounds['bottom'] - (perimeter_pos - (2 * perimeter_width + perimeter_height))
                rot = np.pi / 2  # Vertical orientation
            
            centers_x.append(x)
            centers_y.append(y)
            rotations.append(rot)
            
            placed_ellipses += 1
            current_pos += 1
        
        # Convert to tensors
        centers_x = torch.tensor(centers_x)
        centers_y = torch.tensor(centers_y)
        
        # Radius based on image size, not square size
        radii = torch.ones(num_ellipses) * (image_size * 0.1)
        ns = torch.ones(num_ellipses) * self.initial_n
        rotations = torch.tensor(rotations)
        
        return centers_x, centers_y, radii, ns, rotations