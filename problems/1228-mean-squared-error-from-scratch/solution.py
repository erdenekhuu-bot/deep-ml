import torch

def mse(pred, target):
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    # TODO
    x=pred.flatten()
    y=target.flatten()
    sums = [(i.item() - j.item()) * (i.item()-j.item()) for i, j in zip(x, y)]
    return sum(sums) / pred.numel()
