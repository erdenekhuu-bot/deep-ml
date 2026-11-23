import torch

def compute_cross_entropy_loss(predicted_probs: torch.Tensor, true_labels: torch.Tensor, epsilon: float = 1e-15) -> float:
    """Compute average cross-entropy loss for multi-class classification.
    
    Args:
        predicted_probs: Tensor of predicted probabilities (batch_size, num_classes)
        true_labels: One-hot encoded true labels (batch_size, num_classes)
        epsilon: Small value for numerical stability
    
    Returns:
        Average cross-entropy loss as a float
    """
    clipped = predicted_probs.clamp(min=epsilon)
    log_probs = torch.log(clipped)
    per_sample_loss = (true_labels * log_probs).sum(dim=1)
    loss = -per_sample_loss.mean()
    return loss.item()