import numpy as np


def matrix_completion(D, n, d, r=20, t_max=100, λ=0.1):
    np.random.seed(0)
    X = np.random.normal(size=(d, r))
    Y = np.random.normal(size=(n, r))

    O = (D != 0).astype(int)
    for t in range(1, t_max):
        for k in range(1, d):
            O_Xk = np.diag(O[:, k])  # Diagonal matrix for column k
            # Update X_k using the formula
            fact = Y.T @ O_Xk @ Y + λ * np.eye(r)
            fact = np.linalg.inv(fact)
            X[k, :] =  D[:, k].T @ Y @ fact
             
        for i in range(1, n):
            O_Yi = np.diag(O[i, :])  # Diagonal matrix for row i
            # Update Y_i using the formula
            fact = X.T @ O_Yi @ X + λ * np.eye(r)
            fact = np.linalg.inv(fact)
            Y[i, :] =  D[i, :] @ X @ fact

    return X, Y

def average_squared_error(D, X, Y):
    """
    Compute the average squared approximation error on the observed entries.

    :param D: (np.ndarray) Original data matrix.
    :param X: (np.ndarray) Learned factorized matrix X.
    :param Y: (np.ndarray) Learned factorized matrix Y.
    :return: (float) Average squared approximation error.
    """

    O = (D != 0).astype(int)

    # Reconstructed matrix
    D_hat = Y @ X.T

    # Element-wise multiplication to keep only observed entries
    error_matrix = O * (D - D_hat)

    # Sum of squared errors
    squared_error = np.sum(error_matrix ** 2)

    # Total number of observed entries
    num_observed = np.sum(O)

    # Average squared error
    avg_squared_error = squared_error / num_observed
    return avg_squared_error


def est_ratings(D, X, Y):
    D_hat = Y @ X.T
    print(D_hat)
