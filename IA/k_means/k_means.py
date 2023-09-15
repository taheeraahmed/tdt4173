import numpy as np
from scipy.spatial.distance import cdist

# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class KMeans:

    def __init__(self, k=3, max_iters=10000, pp=False, n_init=1, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.pp = pp
        self.n_init = n_init
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
        """
        X = np.array(X)
        # TODO: Standardize the data
        best_centroids = None
        best_labels = None
        best_distortion = np.inf

        for _ in range(self.n_init):
            if self.pp:
                centroids = X[np.random.choice(len(X), self.k, replace=False)]
            else:
                centroids = self.initialize_centroids(X)

            for _ in range(self.max_iters):
                distances = cdist(X, centroids, 'euclidean')
                labels = np.argmin(distances, axis=1)
                new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

                if np.all(np.abs(new_centroids - centroids) < self.tol):
                    break

                centroids = new_centroids

            distortion = np.sum([np.sum(cdist(X[labels == i], [centroids[i]], 'euclidean')**2) for i in range(self.k)])

            if distortion < best_distortion:
                best_distortion = distortion
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels

    def initialize_centroids(self, X):
        """
        Initialize centroids using K-means++ initialization

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A numpy array of shape (k, n) with initial centroids
        """
        centroids = np.empty((self.k, X.shape[1]))
        centroids[0] = X[np.random.choice(len(X))]  # Choose the first centroid randomly

        for i in range(1, self.k):
            distances = cdist(X, centroids[:i], 'euclidean')
            min_distances = np.min(distances, axis=1)

            # Choose the next centroid with probability proportional to the squared distance
            probabilities = min_distances ** 2 / np.sum(min_distances ** 2)
            new_centroid_idx = np.random.choice(len(X), p=probabilities)
            centroids[i] = X[new_centroid_idx]

        return centroids

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)

        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X_cols = X.columns.tolist()
        distances = cross_euclidean_distance(X[X_cols].values, self.centroids)
        labels = np.argmin(distances, axis=-1)
        return labels

    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm

        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids


# --- Some utility functions

def min_max_scaling(data, feature_range=(0, 1)):
    min_val = feature_range[0]
    max_val = feature_range[1]
    
    min_data = np.min(data, axis=0)
    max_data = np.max(data, axis=0)
    
    scaled_data = (data - min_data) / (max_data - min_data) * (max_val - min_val) + min_val
    return scaled_data

def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 

    Note: by passing "y=0.0", it will compute the euclidean norm

    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 

    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)


def cross_euclidean_distance(x, y=None):
    """
    Calculates the cross pairwise Euclidean distance between two 
    sets of points represented by multidimensional arrays x and y.

    """
    y = x if y is None else y
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion

    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)

        # Check that Xc and mu have compatible shapes for subtraction.
        assert Xc.shape[1] == mu.shape[0], "Xc and mu shapes do not match."
        squared_diff = (Xc - mu) ** 2

        # Sum the squared differences along both axes and then sum the result
        cluster_distortion = squared_diff.sum(axis=1).sum()
        distortion += cluster_distortion

    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 

    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)

    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments

    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]

    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)

    # Intra distance
    a = D[np.arange(len(X)), z]
    # Smallest inter distance
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)

    return np.mean((b - a) / np.maximum(a, b))