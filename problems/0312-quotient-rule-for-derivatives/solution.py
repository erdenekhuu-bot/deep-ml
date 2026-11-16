import torch

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> torch.Tensor:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x) as a scalar torch.Tensor
    """
    gx=torch.tensor(0,dtype=torch.float)
    hx=torch.tensor(0,dtype=torch.float)
    x=torch.tensor(x,dtype=torch.float,requires_grad=True)
    for i in g_coeffs:
        gx=gx*x+i
    for i in h_coeffs:
        hx=hx*x+i
    f=gx/hx
    f.backward()
    return x.grad