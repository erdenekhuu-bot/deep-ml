import torch

def label_encode_ordinal(values: list, order: list) -> torch.Tensor:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        torch.Tensor of integers (dtype=torch.long) representing the encoded values,
        with -1 for any value not found in order
    """
    label_mapping = {label: idx for idx, label in enumerate(order)}
    encoded_values = [label_mapping.get(value, -1) for value in values]
    return torch.tensor(encoded_values, dtype=torch.long)