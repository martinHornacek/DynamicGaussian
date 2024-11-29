import torch
from models.ModelConfig import ModelConfig
from typing import Dict

class MultiSuperellipseModel(torch.nn.Module):
    def __init__(self, config: ModelConfig, image_size: int, square_size: int, x_grid: torch.Tensor, y_grid: torch.Tensor):
        super().__init__()
        self.config = config
        self.num_ellipses = config.num_ellipses
        self.image_size = image_size
        self.square_size = square_size
        self.x_grid = x_grid
        self.y_grid = y_grid
        
        # Initialize parameters using the selected strategy
        centers_x, centers_y, radii, ns, rotations = config.initialization_strategy.initialize(
            config.num_ellipses, image_size
        )
        
        # Create parameters
        self.centers_x = torch.nn.Parameter(centers_x)
        self.centers_y = torch.nn.Parameter(centers_y)
        self.radii = torch.nn.Parameter(radii)
        
        if config.enable_n_parameter:
            self.ns = torch.nn.Parameter(ns)
        else:
            self.register_buffer('ns', torch.ones(config.num_ellipses) * config.fixed_n_value)
            
        if config.enable_rotation:
            self.rotations = torch.nn.Parameter(rotations)
        else:
            self.register_buffer('rotations', torch.zeros(config.num_ellipses))
    
    
    def get_single_superellipse(self, idx):
        eps = 1e-6
        
        x_centered = self.x_grid - self.centers_x[idx]
        y_centered = self.y_grid - self.centers_y[idx]
        
        if self.config.enable_rotation:
            angle = self.rotations[idx]
            x_rot = x_centered * torch.cos(angle) + y_centered * torch.sin(angle)
            y_rot = -x_centered * torch.sin(angle) + y_centered * torch.cos(angle)
        else:
            x_rot = x_centered
            y_rot = y_centered
        
        safe_radius = torch.clamp(self.radii[idx], min=eps)
        x_term = torch.abs(x_rot) / safe_radius
        y_term = torch.abs(y_rot) / safe_radius
        
        x_term = torch.clamp(x_term, max=100)
        y_term = torch.clamp(y_term, max=100)
        
        n_value = self.ns[idx] if self.config.enable_n_parameter else self.config.fixed_n_value
        n_value = torch.clamp(n_value, max=50)
        
        power_sum = torch.clamp(
            x_term ** n_value + y_term ** n_value,
            max=100
        )
        
        result = torch.sigmoid(-5 * (power_sum - 1))
        return 1 - result

    def get_penalties(self) -> Dict[str, torch.Tensor]:
        """Calculate all penalties separately"""
        penalties = {}
        
        # Overlap penalty
        if self.config.w_overlap > 0:
            total_overlap = 0
            for i in range(self.num_ellipses):
                shape_i = self.get_single_superellipse(i)
                for j in range(i + 1, self.num_ellipses):
                    shape_j = self.get_single_superellipse(j)
                    intersection = (1 - shape_i) * (1 - shape_j)
                    total_overlap += torch.sum(intersection) / (self.image_size * self.image_size)
            penalties['overlap'] = total_overlap
        
        # N-parameter regularization
        if self.config.w_n_regularization > 0 and self.config.enable_n_parameter:
            penalties['n_regularization'] = torch.mean(1.0 / (self.ns + 1e-6))
        
        # Radius regularization
        if self.config.w_radius > 0:
            penalties['radius'] = torch.mean(self.radii**2) / (self.square_size**2)
            
        return penalties

    def forward(self):
        image = torch.ones_like(self.x_grid, dtype=torch.float32)
        for i in range(self.num_ellipses):
            superellipse = self.get_single_superellipse(i)
            image = torch.min(image, superellipse)
        return image
