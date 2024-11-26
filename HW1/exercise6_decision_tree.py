import numpy as np

#6a
def gini_impurity(targets):
    counts = np.bincount(targets)
    probabilities = counts / len(targets)
    return 1 - np.sum(probabilities**2)

#6b
def split_cost(L0_y, L1_y, L_y):
    gini_L0 = gini_impurity(L0_y)
    gini_L1 = gini_impurity(L1_y)
    gini_L = gini_impurity(L_y)
    N0, N1, N = len(L0_y), len(L1_y), len(L_y)
    return (N0 / N) * gini_L0 + (N1 / N) * gini_L1 - gini_L