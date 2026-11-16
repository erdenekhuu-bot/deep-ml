import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    if norm_type == 'frobenius' and arr.ndim != 2:
        raise ValueError('Frobenius norm requires 2d')
    arr=arr.flatten()
    if norm_type =='l1':
        return abs(arr).sum().item()
    elif norm_type == 'l2':
        return torch.sqrt(arr.pow(2).sum()).item()
    elif norm_type=='linf':
        return torch.max(abs(arr)).item()
    elif norm_type=='frobenius':
        return torch.sqrt(arr.pow(2).sum()).item()
    else:
        raise ValueError('Unknown norm type')
    # Your code here
    
