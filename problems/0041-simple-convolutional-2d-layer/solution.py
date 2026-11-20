import torch
import torch.nn.functional as F

def simple_conv2d(input_matrix: torch.Tensor, kernel: torch.Tensor, padding: int, stride: int) -> torch.Tensor:
    """
    Perform a 2D convolution on a single-channel input using PyTorch's built-in conv2d.
    input_matrix: 2D tensor (H, W)
    kernel: 2D tensor (kH, kW)
    padding: int, zero-padding on all sides
    stride: int, stride of the convolution
    """
    matrix=input_matrix.unsqueeze(0).unsqueeze(0) 
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    out = F.conv2d(matrix, kernel, padding=padding,stride=stride)
    return out.squeeze(0).squeeze(0)
