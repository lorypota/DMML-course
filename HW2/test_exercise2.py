from exercise2_k_means import *
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
import scipy.sparse.linalg

#2b
dataID, D, labels, r = generateBlobs(epsilon=0.05, n=500)

X_init = init_centroids_greedy_pp(D, r=3, l=10)
X, Y = kmeans(D, r, X_init)

# Decode the cluster assignments from Y matrix
cluster_labels = np.argmax(Y, axis=1)

# Compute the mean approximation error
mean_error = mean_approximation_error(D, X, Y)
print(f"Mean approximation error: {mean_error}")

# Evaluate clustering using Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(labels, cluster_labels)
print(f"Normalized Mutual Information (NMI) score: {nmi_score}")

fig = plt.figure()
ax = plt.axes()
ax.axis('equal')
ax.scatter(D[:, 0], D[:, 1], c=np.argmax(Y,axis=1), s=10)
ax.scatter(X_init.T[:, 0], X_init.T[:, 1], c='red', s=50, marker = 'D') # initial centroids are in red
ax.scatter(X.T[:, 0], X.T[:, 1], c='blue', s=50, marker = 'D') # computed centroids are in blue
plt.show()

#2c
dataID, D, labels, r = generateMoons(epsilon=0.05, n=500)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, kNN in enumerate([5, 10, 20, 40]):
    # Implement here the computation of W as knn graph
    W = kneighbors_graph(D, n_neighbors=kNN, include_self=False)

    rng = np.random.default_rng(seed=0)
    L = np.diag(np.array(W.sum(0))[0]) - W
    v0 = rng.random(min(L.shape))
    Lambda, V = scipy.sparse.linalg.eigsh(L, k=r+1, which="SM", v0=v0)
    A = V[:, 1:]  # remove the first eigenvector, assuming that the graph is conected

    initial_points = init_centroids_greedy_pp(A, r)
    axes[i].scatter(
        initial_points.T[:, 0], initial_points.T[:, 1], c='red', s=50, marker='D', label='Centroids'
    )

    # Perform spectral clustering
    Y = spectral_clustering(W, r, init_centroids_greedy_pp)
    cluster_labels = np.argmax(Y, axis=1)

    # Calculate NMI
    nmi_score = normalized_mutual_info_score(labels, cluster_labels)
    print(f"NMI for kNN={kNN}: {nmi_score}")

    # Scatter plot for clusters
    axes[i].scatter(D[:, 0], D[:, 1], c=cluster_labels, s=10, cmap='viridis')
    axes[i].set_title(f'{dataID}, kNN={kNN}\nNMI={nmi_score:.2f}')
    axes[i].legend()

plt.tight_layout()
plt.show()
