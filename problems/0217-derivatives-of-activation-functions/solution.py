import torch

def activation_derivatives(x: float) -> dict[str, float]:
    """
    Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x
    using PyTorch autograd.
    
    Args:
        x: Input value
        
    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Your code here - use autograd!
    # Hint: Create tensors with requires_grad=True, apply activation, call .backward()
    sigx = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    sig = 1 / (1 + torch.exp(-sigx))
    sig.backward()
    tanhx = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    tanh = (torch.exp(tanhx) - torch.exp(-tanhx)) / (torch.exp(tanhx) + torch.exp(-tanhx))
    tanh.backward()
    relux = torch.tensor(x, requires_grad=True, dtype=torch.float32)
    relu = torch.relu(relux)
    relu.backward()
    return {'sigmoid':sigx.grad.item(),'tanh':tanhx.grad.item(),'relu':relux.grad.item()}