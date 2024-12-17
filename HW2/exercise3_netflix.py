import numpy as np


def matrix_completion(D, r, n, d, t_max=100, λ=0.1):
    # 1. init random X, Y
    np.random.seed(0)
    X = np.random.normal(size=(d, r))
    Y = np.random.normal(size=(n, r))
    
    # 2. indicator non zero
    

    return X, Y