import torch

def min_max(x: torch.Tensor) -> torch.Tensor:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A tensor of numerical values
    
    Returns:
        A new tensor with values normalized to [0, 1]
    """
    # Your code here
    return (x - x.min()) / (x.max() - x.min())