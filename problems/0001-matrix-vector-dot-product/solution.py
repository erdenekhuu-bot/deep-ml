import numpy as np

def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

	try:
        return np.dot(a,b)
    except Exception:
        return -1