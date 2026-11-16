import torch

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient (float)
        - direction: Unit vector (torch.Tensor) in direction of steepest ascent
        - descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
    """
    # Your code here
    gradient=torch.tensor(gradient,dtype=float)
    if (gradient == 0).all():
        return {
            'magnitude': 0.0, 
            'direction': torch.zeros_like(gradient), 
            'descent_direction': torch.zeros_like(gradient)
        }
    
    l2=torch.sqrt(gradient.pow(2).sum()).item()
    dir=gradient/l2
    inv_dir=dir*-1
    return {'magnitude':l2, 'direction':dir,'descent_direction':inv_dir}