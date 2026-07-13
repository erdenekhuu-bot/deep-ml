import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	return np.mean(matrix,axis=1 if mode=='row' else 0)