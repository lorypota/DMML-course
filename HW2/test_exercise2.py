from exercise2_k_means import *
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph
import matplotlib.pyplot as plt

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

# fig = plt.figure()
# ax = plt.axes()
# ax.axis('equal')
# ax.scatter(D[:, 0], D[:, 1], c=np.argmax(Y,axis=1), s=10)
# ax.scatter(X_init.T[:, 0], X_init.T[:, 1], c='red', s=50, marker = 'D') # initial centroids are in red
# ax.scatter(X.T[:, 0], X.T[:, 1], c='blue', s=50, marker = 'D') # computed centroids are in blue
# plt.show()

#2c
dataID, D, labels, r = generateMoons(0.05,n=500)

for kNN in [15, 25, 30, 35]:
    # Implement here the computation of W as knn graph
    # W = radius_neighbors_graph(D,0.5,include_self=False)
    W = kneighbors_graph(D, n_neighbors=kNN, mode='connectivity', include_self=False).toarray()
    X_init = init_centroids_greedy_pp(D, r=3, l=10)
    Y = spectral_clustering(W,r,X_init)
    cluster_labels = np.argmax(Y, axis=1)
    nmi_score = normalized_mutual_info_score(labels, cluster_labels)
    print(f"NMI for kNN={kNN}: {nmi_score}")

    plt.scatter(D[:, 0], D[:, 1], c=np.argmax(Y,axis=1), s=10)
    plt.title('%s'  % ( dataID) )
    plt.show()