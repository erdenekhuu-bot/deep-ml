import torch
from typing import List

def log_softmax(scores: List[float]) -> torch.Tensor:
    """
    Compute the log-softmax of a 1D list of scores using PyTorch.
    Args:
        scores: list of floats
    Returns:
        torch.Tensor of log-softmax values
    """
    x=torch.tensor(scores,dtype=torch.float32)
    max_val=x.max()
    shifted=x-max_val
    log_sum_exp=torch.log(torch.exp(shifted).sum())
    return shifted-log_sum_exp
