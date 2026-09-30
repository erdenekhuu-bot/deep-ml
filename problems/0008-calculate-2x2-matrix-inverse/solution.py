import torch

def inverse_2x2(matrix) -> torch.Tensor | None:
    """
    Returns:
        A 2x2 tensor containing the inverse, or None if the matrix is singular
    """
    m = torch.as_tensor(matrix, dtype=torch.float)
    # Your code here
    try:
        return torch.linalg.inv(m)
    except Exception:
        return None