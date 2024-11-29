import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Dict, List
from models.ModelConfig import ModelConfig

def create_animation(
    ground_truth: np.ndarray,
    frames: List[np.ndarray],
    metrics: Dict[str, List[float]],
    config: ModelConfig,
    save_path: str = 'superellipse_optimization.gif'
):
    """
    Create and save animation showing optimization progress and metrics
    """
    def safe_log_scale_limits(values, buffer=0.1):
        """Calculate safe limits for log scale plots, handling zeros and negatives"""
        min_val = min(values)
        max_val = max(values)
        
        if min_val <= 0:
            # Find the smallest positive value
            positive_vals = [v for v in values if v > 0]
            if positive_vals:
                min_val = min(positive_vals)
            else:
                # If no positive values, use a small positive number
                min_val = 1e-10
        
        # Add buffer for better visualization
        log_min = np.log10(min_val)
        log_max = np.log10(max(max_val, min_val * 1.1))  # Ensure max > min
        
        return 10 ** (log_min - buffer), 10 ** (log_max + buffer)
    
    # Calculate number of metric plots needed (excluding mse_loss and total_loss)
    n_penalty_metrics = len(metrics) - 2  # Subtract mse_loss and total_loss
    
    # Create figure layout
    fig = plt.figure(figsize=(18, 6 + (n_penalty_metrics > 3) * 4))
    
    if n_penalty_metrics <= 3:
        gs = plt.GridSpec(2, 3)
    else:
        gs = plt.GridSpec(3, 3)
    
    # Ground truth subplot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ground_truth, cmap='gray')
    ax1.set_title("Ground Truth")
    
    # Optimization progress subplot
    ax2 = fig.add_subplot(gs[0, 1])
    img_display = ax2.imshow(frames[0], cmap='gray')
    ax2.set_title("Optimization Progress")
    epoch_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                         color='red', fontsize=10, verticalalignment='top')
    
    # Initialize metric plots
    metric_lines = {}
    
    # First plot: reconstruction loss and total loss
    ax3 = fig.add_subplot(gs[0, 2])
    metric_lines['total_loss'], = ax3.plot([], [], 'r-', label='Total Loss')
    metric_lines['mse_loss'], = ax3.plot([], [], 'b-', label='Reconstruction Loss')
    ax3.set_xlim(0, config.num_epochs)
    
    # Safe log scale limits for loss plot
    combined_losses = metrics['total_loss'] + metrics['mse_loss']
    y_min, y_max = safe_log_scale_limits(combined_losses)
    ax3.set_ylim(y_min, y_max)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.set_title("Loss Curves")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    
    # Additional penalty plots
    penalty_metrics = [k for k in metrics.keys() 
                      if k not in ['total_loss', 'mse_loss']]
    
    for i, metric_name in enumerate(penalty_metrics):
        row = 1 + i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])
        metric_lines[metric_name], = ax.plot([], [], label=metric_name)
        ax.set_xlim(0, config.num_epochs)
        
        # Safe log scale limits for each penalty metric
        y_min, y_max = safe_log_scale_limits(metrics[metric_name])
        ax.set_ylim(y_min, y_max)
        ax.set_yscale('log')
        ax.legend()
        ax.set_title(f"{metric_name} over epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
    
    plt.tight_layout()
    
    def update(frame):
        # Update image
        img_display.set_array(frames[frame])
        current_epoch = frame * config.frame_interval
        epoch_text.set_text(f'Epoch: {current_epoch}')
        
        # Update all metric plots
        frame_idx = current_epoch + 1
        for metric_name, line in metric_lines.items():
            data = metrics[metric_name][:frame_idx]
            # Handle zero values by replacing them with minimum positive value
            min_positive = min([x for x in data if x > 0]) if any(x > 0 for x in data) else 1e-10
            data = [max(x, min_positive) for x in data]
            line.set_data(range(frame_idx), data)
        
        return [img_display, epoch_text] + list(metric_lines.values())
    
    # Create animation
    anim = FuncAnimation(
        fig, update,
        frames=len(frames),
        interval=50,
        blit=True
    )
    
    # Save animation
    anim.save(save_path, writer='pillow')
    plt.close()

def plot_results(
    ground_truth: np.ndarray,
    frames: List[np.ndarray],
    metrics: Dict[str, List[float]]
):
    """Plot training results with all tracked metrics"""
    def safe_log_scale_limits(values, buffer=0.1):
        """Calculate safe limits for log scale plots, handling zeros and negatives"""
        min_val = min(values)
        max_val = max(values)
        
        if min_val <= 0:
            # Find the smallest positive value
            positive_vals = [v for v in values if v > 0]
            if positive_vals:
                min_val = min(positive_vals)
            else:
                # If no positive values, use a small positive number
                min_val = 1e-10
        
        # Add buffer for better visualization
        log_min = np.log10(min_val)
        log_max = np.log10(max(max_val, min_val * 1.1))  # Ensure max > min
        
        return 10 ** (log_min - buffer), 10 ** (log_max + buffer)
    
    # Setup main figure with all metrics
    n_metrics = len(metrics)
    fig = plt.figure(figsize=(15, 5 + (n_metrics - 1) * 2))
    
    # Top row: Ground truth, final result, and first metric plot
    gs = plt.GridSpec(2 + (n_metrics-1)//2, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(ground_truth, cmap='gray')
    ax1.set_title("Ground Truth")
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(frames[-1], cmap='gray')
    ax2.set_title("Final Result")
    
    # Plot all metrics
    for i, (name, values) in enumerate(metrics.items()):
        row = (i + 2) // 3
        col = (i + 2) % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Handle zero values by replacing them with minimum positive value
        min_positive = min([x for x in values if x > 0]) if any(x > 0 for x in values) else 1e-10
        plot_values = [max(x, min_positive) for x in values]
        
        ax.plot(plot_values, label=name)
        ax.set_title(f"{name} over epochs")
        
        # Set y-axis limits safely
        y_min, y_max = safe_log_scale_limits(values)
        ax.set_ylim(y_min, y_max)
        ax.set_yscale('log')
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend()
    
    plt.tight_layout()
    plt.show()