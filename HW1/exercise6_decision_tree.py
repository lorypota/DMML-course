import numpy as np

#6a
def gini_impurity(targets):
    counts = np.bincount(targets)
    probabilities = counts / len(targets)
    return 1 - np.sum(probabilities**2)

#6b
