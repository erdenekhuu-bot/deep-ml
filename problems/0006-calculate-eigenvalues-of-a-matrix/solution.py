import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    trace = torch.diag(matrix).sum()
    det = torch.linalg.det(matrix)
    
    dis = trace * trace - 4 * det
    
    eig1 = (trace + torch.sqrt(dis)) / 2
    eig2 = (trace - torch.sqrt(dis)) / 2
    
    return torch.tensor([eig1, eig2], dtype=matrix.dtype)
    
