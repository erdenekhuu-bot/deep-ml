import math
import numpy as np

def sigmoid(z: float) -> float:
    #Your code here
    return 1/(1+np.exp(-z))