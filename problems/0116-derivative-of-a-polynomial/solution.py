import torch

def poly_term_derivative(c: float, x: float, n: float) -> torch.Tensor:
    """
    Compute the derivative of a polynomial term c * x^n at point x.
    
    Args:
        c: coefficient of the term
        x: point at which to evaluate the derivative
        n: exponent of the term
    
    Returns:
        The value of the derivative at point x as a tensor
    """
    x=torch.tensor(x,requires_grad=True)
    y=c*x**n
    y.backward()
    return x.grad
    