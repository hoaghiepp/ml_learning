import numpy as np
from clustering_package.model.base import BaseClusteringAlgorithm


def kmeans_plusplus(X, n_clusters, sample_weight=None):
    """
    Initialize cluster centers using the KMeans++ algorithm.

    Parameters:
        X (ndarray): Dataset of shape (n_samples, n_features)
        n_clusters (int): Number of clusters to form
        sample_weight (ndarray or None): Sample weights for each observation X
    Returns:
        centers (ndarray): Initialized cluster centers, shape (n_clusters, n_features)
    """
    n_samples, n_features = X.shape
    centroids = np.empty((n_clusters, n_features))

    if sample_weight is not None:
        probabilities = sample_weight / np.sum(sample_weight)
        first_centroid_idx = np.random.choice(n_samples, p=probabilities)
    else:
        first_centroid_idx = np.random.choice(n_samples)

    centroids[0] = X[first_centroid_idx]

    for i in range(1, n_clusters):
        distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
        # Compute probabilities proportional to D(x)^2, weighted by sample_weight if provided
        if sample_weight is not None:
            weighted_distances_sq = distances ** 2 * sample_weight
            probabilities = weighted_distances_sq / np.sum(weighted_distances_sq)
        else:
            probabilities = distances ** 2 / np.sum(distances ** 2)

        next_centroid_idx = np.random.choice(n_samples, p=probabilities)
        centroids[i] = X[next_centroid_idx]
    return centroids

class KMeans(BaseClusteringAlgorithm):
    """
    K-Means clustering algorithm.

    Parameters
    ----------
    n_clusters : int, default=8
        The number of clusters to form as well as the number of centroids to generate.
    init: {'k-means++', 'random'}, default='k-means++'
        Method for initialization
    n_init: ‘auto’ or int, default=’auto’
        Number of times the k-means algorithm is run with different centroid seeds.
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

    def __init__(self, n_clusters=8, init='k-means++', n_init='auto', max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

        if self.init not in ('k-means++', 'random'):
            raise ValueError("init must be 'k-mean++' or 'random'")

    def _initialize_centroids(self, X, sample_weight=None, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        if self.init == 'random':
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]

        elif self.init == 'k-means++':
            return kmeans_plusplus(X, n_clusters=self.n_clusters, sample_weight=sample_weight)

    def _assign_clusters(self, X, centroids):
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels, sample_weight=None):
        new_centroids = np.empty((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            points_in_cluster = X[labels == i]
            weights_in_cluster = sample_weight[labels == i] if sample_weight is not None else None

            if len(points_in_cluster) > 0:
                if weights_in_cluster is not None and np.sum(weights_in_cluster) > 0:
                    new_centroids[i] = np.average(points_in_cluster, axis=0, weights=weights_in_cluster)
                else:
                    new_centroids[i] = np.mean(points_in_cluster, axis=0)
            else:
                # if a cluster becomes empty, keeps its centroid in place
                new_centroids[i] = self.cluster_centers_[i]
        return new_centroids

    def _calculate_inertia(self, X, labels, centroids, sample_weight=None):
        inertia = 0
        for i in range(self.n_clusters):
            points_in_cluster = X[labels == i]
            if len(points_in_cluster) > 0:
                distances = np.linalg.norm(X - centroids[i], axis=1) ** 2
                if sample_weight is not None:
                    weights_in_cluster = sample_weight[labels == i]
                    inertia += np.sum(distances * weights_in_cluster)
                else:
                    inertia += np.sum(distances)
        return inertia

    def fit(self, X, y=None, sample_weight=None):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations are assigned
            equal weight.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = np.asarray(X, dtype=np.float64)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError("sample_weight must have the same number of samples as X.")

        best_inertia = np.inf
        best_labels = None
        best_cluster_centers = None

        if self.n_init == 'auto':
            self.n_init = 1

        for run in range(self.n_init):
            current_random_state = self.random_state + run if self.random_state is not None else None

            # Initialize centroids for the current run
            current_cluster_centers = self._initialize_centroids(X, sample_weight, current_random_state)
            current_labels = None

            for i in range(self.max_iter):
                current_labels = self._assign_clusters(X, current_cluster_centers)
                new_centroids = self._update_centroids(X, current_labels, sample_weight)

                if np.linalg.norm(new_centroids - current_cluster_centers) < self.tol:
                    break
                current_cluster_centers = new_centroids
            current_inertia = self._calculate_inertia(X, current_labels, current_cluster_centers, sample_weight)

            if current_inertia < best_inertia:
                best_inertia = current_inertia
                best_labels = current_labels
                best_cluster_centers = current_cluster_centers

        self.cluster_centers_ = best_cluster_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

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

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            Distance of each sample to each cluster center.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("KMeans model has not been fitted yet. Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        return np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2)).T

    def fit_transform(self, X, y=None, sample_weight=None):
        """
        Compute clustering and transform X to cluster-distance space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations are assigned
            equal weight.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            Distance of each sample to each cluster center.
        """
        self.fit(X, y=y, sample_weight=sample_weight)
        return self.transform(X)

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations are assigned
            equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        self.fit(X, y=y, sample_weight=sample_weight)
        return self.predict(X)
