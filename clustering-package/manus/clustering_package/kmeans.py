import numpy as np
from clustering_package.base import BaseClusteringAlgorithm

class KMeans(BaseClusteringAlgorithm):
    """
    K-Means clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a single run.
    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers
        of two consecutive iterations to declare convergence.
    random_state : int, default=None
        Determines random number generation for centroid initialization.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def _initialize_centroids(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else self.cluster_centers_[i] for i in range(self.n_clusters)])
        return new_centroids

    def fit(self, X, y=None):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=np.float64)
        self.cluster_centers_ = self._initialize_centroids(X)

        for i in range(self.max_iter):
            self.labels_ = self._assign_clusters(X, self.cluster_centers_)
            new_centroids = self._update_centroids(X, self.labels_)

            if np.linalg.norm(new_centroids - self.cluster_centers_) < self.tol:
                break
            self.cluster_centers_ = new_centroids
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return self._assign_clusters(X, self.cluster_centers_)


