import numpy as np
import sklearn
import sklearn.datasets
from sklearn.metrics import normalized_mutual_info_score

def RSS(D,X,Y):
    return np.sum((D- Y@X.T)**2)
def getY(labels):
    '''
        Compute the cluster assignment matrix Y from the categorically encoded labels
    '''
    Y = np.eye(max(labels)+1)[labels]
    return Y
def update_centroid(D,Y):
    cluster_sizes = np.diag(Y.T@Y).copy()
    cluster_sizes[cluster_sizes==0]=1
    return D.T@Y/cluster_sizes
def update_assignment(D,X):
    dist = np.sum((np.expand_dims(D,2) - X)**2,1)
    labels = np.argmin(dist,1)
    return getY(labels)
def kmeans(D,r, X_init, epsilon=0.00001, t_max=10000):
    X = X_init.copy()
    Y = update_assignment(D,X)
    rss_old = RSS(D,X,Y) +2*epsilon
    t=0
    #Looping as long as difference of objective function values is larger than epsilon
    while rss_old - RSS(D,X,Y) > epsilon and t < t_max-1:
        rss_old = RSS(D,X,Y)
        X = update_centroid(D,Y)
        Y = update_assignment(D,X)
        t+=1
    print(t,"iterations")
    return X,Y


def generateMoons(epsilon, n):
    moons, labels = sklearn.datasets.make_moons(n_samples=n, noise=epsilon, random_state=7)
    return "moons", moons, labels, 2
def generateBlobs(epsilon, n):
    blobs, labels = sklearn.datasets.make_blobs(n_samples=n,centers=3, cluster_std=[epsilon + 1, epsilon + 1.5, epsilon + 0.5], random_state=54)
    return "blobs", blobs, labels, 3


def init_centroids_greedy_pp(D,r,l=10):
    '''
        :param r: (int) number of centroids (clusters)
        :param D: (np-array) the data matrix
        :param l: (int) number of centroid candidates in each step
        :return: (np-array) 'X' the selected centroids from the dataset
    '''   
    rng =  np.random.default_rng(seed=7) # use this random generator to sample the candidates (sampling according to given probabilities can be done via rng.choice(..))
    n,d = D.shape

    indexes = rng.integers(low=0, high=n, size=r)
    X = np.array(D[indexes,:]).T
    return X


dataID, D, labels, r = generateBlobs(epsilon=0.05, n=500)

X_init = init_centroids_greedy_pp(D,r=3,l=10)
X,Y = kmeans(D,r, X_init)

# Decode the cluster assignments from Y matrix
cluster_labels = np.argmax(Y, axis=1)


def mean_approximation_error(D, X, Y):
    """
    Computes the mean approximation error for k-means clustering.

    :param D: (np.ndarray) Data points of shape (n, d)
    :param X: (np.ndarray) Centroids of shape (d, k)
    :param Y: (np.ndarray) Cluster assignment matrix of shape (n, k)
    :return: (float) Mean approximation error
    """
    n, d = D.shape
    rss = RSS(D, X, Y)  # Residual sum of squares
    return rss / (n * d)

# Compute the mean approximation error
mean_error = mean_approximation_error(D, X, Y)
print(f"Mean approximation error: {mean_error}")

# Evaluate clustering using Normalized Mutual Information (NMI)
nmi_score = normalized_mutual_info_score(labels, cluster_labels)
print(f"Normalized Mutual Information (NMI) score: {nmi_score}")


def distance(v, X):
    min = float('inf')
    for x in X.T:
        dist = np.linalg.norm(v - x)**2
        if dist < min:
            min = dist
    return min


def greedy_k_means(data, r, l):
    # Sample i_1, ..., i_l in {1, ..., n} uniformly at random
    n = data.shape[0]
    random_indices = np.random.choice(n, l, replace=False)

    # i <- arg min i in {i_1, ..., i_l} sum_{j=1}^n ||x_j - x_i||^2
    min = float('inf')
    i_min = -1
    for random_index in random_indices:
        sum = 0
        for j in range(n):
            sum += np.linalg.norm(data[j] - data[random_index])**2
        if sum < min:
            min = sum
            i_min = random_index

    # X <- D^T
    X = data[i_min].T

    # s <- 2
    s = 2

    while s <= r:
        # Calculate propabilities p_i
        sum = 0
        for j in range(n):
            sum += distance(data[j], X)

        probabilities = []
        for i in range(n):
            probabilities[i] = distance(data[i], X) / sum

        # Sample i_1, ..., i_l in {1, ..., n} independently with probability p_i
        indices_with_probability = np.random.choice(n, l, p=probabilities)

        # i <- arg min i in {i_1, ..., i_l} sum_{j=1}^n dist(D_j, [X | D_i^T])
        X_temp = None
        X_temp_min = None
        min_value = float('inf')
        for random_index in indices_with_probability:
            X_temp = np.concatenate((X, data[random_index].T), axis=1)
            sum = 0
            for j in range(n):
                sum += distance(data[j], X_temp)
            if sum < min_value:
                min_value = sum
                X_temp_min = X_temp
        X = X_temp_min
        s += 1
    return X