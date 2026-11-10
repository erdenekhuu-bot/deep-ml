import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    tensor=torch.tensor(scores)
    exp=torch.exp(tensor-tensor.max())
    sumexp=torch.exp(tensor-tensor.max()).sum()
    result=exp/sumexp
    return result.tolist()
