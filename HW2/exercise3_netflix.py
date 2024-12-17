import numpy as np


def matrix_completion(D, r, n, d, t_max=100, λ=0.1):
    np.random.seed(0)
    X = np.random.normal(size=(d, r))
    Y = np.random.normal(size=(n, r))
    # Implement now the optimization procedure
    return X, Y