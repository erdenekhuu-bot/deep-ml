import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    B=A.T @ A
    theta=0
    if np.diag(B)[0] == np.diag(B)[1]:
        theta=np.pi / 4
    else:
        theta = np.arctan2(
            2 * B[0, 1],
            B[0, 0] - B[1, 1]
        ) / 2
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    jacob_rot = R.T @ B @ R
    eigenvalues = np.diag(jacob_rot)
    s_values = np.sqrt(eigenvalues)
    V = R
    indices = np.argsort(s_values)[::-1]
    s_values = s_values[indices]
    V = V[:, indices]
    sigma = np.diag(s_values)
    sigma_inv = np.linalg.inv(sigma)
    U = A @ V @ sigma_inv
    return U,s_values,V.T