import numpy as np
import torch
import torch.optim as optim

from typing import Dict, List, Tuple
from models.ModelConfig import ModelConfig
from models.MultiSuperellipseModel import MultiSuperellipseModel
from helpers.plotting import create_animation, plot_results

def train_and_collect_frames(
    model: MultiSuperellipseModel,
    ground_truth_tensor: torch.Tensor,
    config: ModelConfig
) -> Tuple[List[np.ndarray], Dict[str, List[float]]]:
    """
    Train the model and collect frames and metrics
    """
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    frames = []
    metrics = {
        'total_loss': [],
        'mse_loss': [],
    }
    
    # Initialize penalty metrics based on non-zero weights
    if config.w_overlap > 0:
        metrics['overlap_penalty'] = []
    if config.w_n_regularization > 0 and config.enable_n_parameter:
        metrics['n_regularization_penalty'] = []
    if config.w_radius > 0:
        metrics['radius_penalty'] = []
    
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        
        output = model()
        mse_loss = torch.nn.functional.mse_loss(output, ground_truth_tensor)
        
        # Get all penalties
        penalties = model.get_penalties()
        
        # Calculate weighted sum of penalties
        total_loss = mse_loss
        for name, penalty in penalties.items():
            weight = getattr(config, f'w_{name}')
            total_loss = total_loss + weight * penalty
        
        # Check for NaN
        if torch.isnan(total_loss):
            print(f"NaN detected at epoch {epoch}")
            break
            
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Apply constraints
        with torch.no_grad():
            model.radii.clamp_(model.square_size/8, model.square_size/2)
            model.centers_x.clamp_(0, model.image_size)
            model.centers_y.clamp_(0, model.image_size)
            if config.enable_n_parameter:
                model.ns.clamp_(2, 50)
            if config.enable_rotation:
                model.rotations.clamp_(-np.pi, np.pi)
        
        # Store metrics
        metrics['total_loss'].append(total_loss.item())
        metrics['mse_loss'].append(mse_loss.item())
        for name, penalty in penalties.items():
            metrics[f'{name}_penalty'].append(penalty.item())
        
        if epoch % config.frame_interval == 0:
            with torch.no_grad():
                frames.append(model().numpy().copy())
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}")
                for name, values in metrics.items():
                    if values:
                        print(f"{name}: {values[-1]:.6f}")
                print()
                
    return frames, metrics

def run_experiment(config: ModelConfig, save_path: str = None):
    """
    Run a complete experiment with the given configuration
    """
    # Setup image size
    image_size = 256
    square_size = 128
    
    # Create ground truth image
    ground_truth = np.ones((image_size, image_size))
    center = image_size // 2
    half_square = square_size // 2
    ground_truth[center - half_square:center + half_square, 
                 center - half_square:center + half_square] = 0
    
    # Convert to tensor
    ground_truth_tensor = torch.FloatTensor(ground_truth)
    
    # Create coordinate grid
    x = torch.linspace(0, image_size-1, image_size)
    y = torch.linspace(0, image_size-1, image_size)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    # Create and train model
    model = MultiSuperellipseModel(config, image_size, square_size, x_grid, y_grid)
    frames, metrics = train_and_collect_frames(model, ground_truth_tensor, config)
    
    # Create animation
    if save_path is None:
        save_path = f'superellipse_optimization_{config.num_ellipses}.gif'
    create_animation(ground_truth, frames, metrics, config, save_path)
    
    # Plot final results
    plot_results(ground_truth, frames, metrics)
    
    # Print configuration used
    print("\nConfiguration used:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")
    
    return model, frames, metrics

if __name__ == "__main__":
    # Configurations for different ablation studies
    configs = {
        "baseline": ModelConfig()
    }

    # Run baseline
    model, frames, metrics = run_experiment(
        configs["baseline"],
        save_path='perimeter-3-10k.gif'
    )