import torch

def solve_jacobi(A, b, n) -> torch.Tensor:
    """
    Solve Ax = b using the Jacobi iterative method for n iterations.
    A: (m,m) tensor; b: (m,) tensor; n: number of iterations.
    Returns a 1-D tensor of length m, rounded to 4 decimals.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    b_t = torch.as_tensor(b, dtype=torch.float)
    mask=~torch.eye(A_t.shape[0],dtype=torch.bool)
    x=torch.zeros(b_t.shape[0])
    diag=A_t.diag()
    nondiag=A_t*mask

    # i saw matrice multiplication without @ sign same as sum() horizontal
    for _ in range(n):
       x = (b_t - (nondiag * x).sum(dim=1)) / diag
        
    return torch.round(x,decimals=4)
