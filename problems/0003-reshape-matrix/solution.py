import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	x, y = new_shape
    arr = np.array(a)
    if arr.size == x * y:
        return arr.reshape(x, y).tolist()
    else:
        return []
