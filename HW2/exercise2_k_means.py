import numpy as np
import sklearn
import sklearn.datasets
from sklearn.metrics import normalized_mutual_info_score
import scipy

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
    rng =  np.random.default_rng(seed=7) # random generator to sample candidates (via rng.choice(..))
    n, d = D.shape

    # Sample i_1, ..., i_l in {1, ..., n} uniformly at random
    random_indices = rng.choice(n, l, replace=False)

    # i <- arg min i in {i_1, ..., i_l} sum_{j=1}^n ||x_j - x_i||^2
    min = float('inf')
    i_min = -1
    for random_index in random_indices:
        sum = 0
        for j in range(n):
            sum += np.linalg.norm(D[j] - D[random_index])**2
        if sum < min:
            min = sum
            i_min = random_index

    # X <- D^T
    X = D[i_min].T.reshape(-1, 1)

    # s <- 2
    s = 2

    while s <= r:
        # Calculate propabilities p_i
        sum = 0
        for j in range(n):
            sum += distance(D[j], X)

        probabilities = []
        for i in range(n):
            probabilities.append(distance(D[i], X) / sum)

        # Sample i_1, ..., i_l in {1, ..., n} independently with probability p_i
        indices_with_probability = rng.choice(n, l, p=probabilities)

        # i <- arg min i in {i_1, ..., i_l} sum_{j=1}^n dist(D_j, [X | D_i^T])
        X_temp = None
        X_temp_min = None
        min_value = float('inf')
        for random_index in indices_with_probability:
            X_temp = np.concatenate((X, D[random_index].T.reshape(-1, 1)), axis=1)
            sum = 0
            for j in range(n):
                sum += distance(D[j], X_temp)
            if sum < min_value:
                min_value = sum
                X_temp_min = X_temp
        X = X_temp_min
        s += 1
    return X


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


def distance(v, X):
    min = float('inf')
    for x in X.T:
        dist = np.linalg.norm(v - x)**2
        if dist < min:
            min = dist
    return min
#c
def spectral_clustering(W,r, X_init):
    '''
        :param W: (np-array) nxn similarity/weighted adjacency matrix
        :param r: (int) number of centroids (clusters)
        :param X_init: (function) the centroid initialization function 
        :return: (np-array) 'Y' the computed cluster assignment matrix
    '''  
    np.random.seed(0)
    L = np.diag(np.array(W.sum(0))[0]) - W
    v0 = np.random.rand(min(L.shape))
    Lambda, V = scipy.sparse.linalg.eigsh(L, k=r+1, which="SM", v0=v0)
    A = V[:,1:] #remove the first eigenvector, assuming that the graph is conected
    initial_points = X_init(A,r,l=10)
    X, Y = kmeans(A, r, initial_points)

    return Y