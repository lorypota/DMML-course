import numpy as np

#5a
def proportional_probability(target, categories):
    results = {}
    counts = np.bincount(target)
    for i, category in enumerate(categories):
        results[category] = counts[i] / len(target)
    return results

#5b
