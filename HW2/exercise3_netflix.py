import numpy as np


def matrix_completion(D, n, d, r=20, t_max=100, λ=0.1):
    # 1. init random X, Y
    np.random.seed(0)
    X = np.random.normal(size=(d, r))
    Y = np.random.normal(size=(n, r))

    # 2. indicator non zero
    O = (D != 0).astype(int)

    # 3. repeat t_max times
    for t in range(t_max):
        # 3.1. update X
        for k in range(d):
            # 3.1.1. Oxk <- diag(O1k, O2k, ..., Onk)
            O_Xk = np.diag(O[:, k])
            # 3.1.2. Xk <- Dk^T * Y * (Y^T * Oxk * Y + λ * I)^-1
            fact = Y.T @ O_Xk @ Y + λ * np.eye(r)
            fact = np.linalg.inv(fact)
            X[k, :] =  D[:, k].T @ Y @ fact
        
        # 3.2. update Y
        for i in range(n):
            # 3.2.1. Oi <- diag(Oi1, Oi2, ..., Oid)
            O_Yi = np.diag(O[i, :])
            # 3.2.2. Yi <- Di * X * (X^T * Oi * X + λ * I)^-1
            fact = X.T @ O_Yi @ X + λ * np.eye(r)
            fact = np.linalg.inv(fact)
            Y[i, :] =  D[i, :] @ X @ fact

    check_3b(O, X, Y)

    return X, Y

def check_3b(O, X, Y):
    # Compute mean of missing entries
    YX_T = Y @ X.T
    missing_entries = YX_T[O == 0]
    print(f"Mean of missing entries: {np.mean(missing_entries)}")

    # Compute the number of missing value imputations outside [0.5, 5]
    count_out_of_range = np.sum((missing_entries < 0.5) | (missing_entries > 5))
    print(f"Number of missing value imputations outside [0.5, 5]: {count_out_of_range}")

    # Compute variance of missing value imputations
    variance_missing = np.var(missing_entries)
    print(f"Variance of missing value imputations: {variance_missing}")


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
    error_matrix = D - (O * D_hat)

    # Sum of squared errors
    squared_error = np.linalg.norm(error_matrix) ** 2

    # Total number of observed entries
    num_observed = np.sum(O)

    # Average squared error
    avg_squared_error = squared_error / num_observed
    return avg_squared_error
    

def find_est_by_movie_id(D, X, Y, filtered_movie_ids, movie_id, user=0):
    (n, d) = D.shape
    assert X.shape[0] == d, "Number of rows in X does not match number of users."
    assert Y.shape[0] == n, "Number of rows in Y does not match number of movies."

    new_movie_id = filtered_movie_ids.get_loc(movie_id)
    
    D_estimated = Y @ X.T

    predicted_rating = D_estimated[user, new_movie_id]

    return predicted_rating
