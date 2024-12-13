from exercise2_k_means import *

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

# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = plt.axes()
# ax.axis('equal')
# ax.scatter(D[:, 0], D[:, 1], c=np.argmax(Y,axis=1), s=10)
# ax.scatter(X_init.T[:, 0], X_init.T[:, 1], c='red', s=50, marker = 'D') # initial centroids are in red
# ax.scatter(X.T[:, 0], X.T[:, 1], c='blue', s=50, marker = 'D') # computed centroids are in blue
# plt.show()