import numpy as np

def proportional_probability(dataset, categories):
    results = {}
    counts = np.bincount(dataset)
    for i, category in enumerate(categories):
        results[category] = counts[i] / len(dataset)
    return results