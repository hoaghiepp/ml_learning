import numpy as np
from collections import deque
from clustering_package.model.base import BaseClusteringAlgorithm

class DBSCAN(BaseClusteringAlgorithm):
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm.

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
        This includes the point itself.
    metric : str, default=\'euclidean\'
        The metric to use when calculating distance between instances in a feature space.
        If metric is \'euclidean\', then Euclidean distance is used.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    components_ : ndarray of shape (n_core_samples, n_features)
        Copy of each core sample found by training.
    """
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None

        if self.metric != 'euclidean':
            raise ValueError("Currently only 'euclidean' metric is supported.")

    def _get_neighbors(self, X, point_idx):
        if self.metric == 'euclidean':
            distances = np.linalg.norm(X - X[point_idx], axis=1)
            return np.where(distances <= self.eps)[0]
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def fit(self, X, y=None, sample_weight=None):
        """
        Perform DBSCAN clustering from features, or distance matrix.

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
        n_samples = X.shape[0]

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != n_samples:
                raise ValueError("sample_weight must have the same number of samples as X.")
        else:
            sample_weight = np.ones(n_samples, dtype=np.float64)

        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1 means noise
        visited = np.zeros(n_samples, dtype=bool)
        core_samples = []
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._get_neighbors(X, i)

            if np.sum(sample_weight[neighbors]) < self.min_samples:
                continue  # Not a core point
            else:
                self._expand_cluster(X, sample_weight, i, neighbors, cluster_id, visited, core_samples)
                cluster_id += 1

        self.core_sample_indices_ = np.array(sorted(core_samples), dtype=int)
        self.components_ = X[self.core_sample_indices_]
        return self

    def _expand_cluster(self, X, sample_weight, point_idx, neighbors, cluster_id, visited, core_samples):
        self.labels_[point_idx] = cluster_id
        queue = deque(neighbors)

        if np.sum(sample_weight[neighbors]) >= self.min_samples:
            core_samples.append(point_idx)

        while queue:
            current_point = queue.popleft()

            if not visited[current_point]:
                visited[current_point] = True
                current_neighbors = self._get_neighbors(X, current_point)

                if np.sum(sample_weight[current_neighbors]) >= self.min_samples:
                    queue.extend(n for n in current_neighbors if n not in queue)

                    if current_point not in core_samples:
                        core_samples.append(current_point)

            if self.labels_[current_point] == -1:
                self.labels_[current_point] = cluster_id

    def predict(self, X):
        """
        Predict the cluster labels for new data.
        Note: DBSCAN does not naturally extend to new data. This implementation assigns
        new data points to the cluster of their nearest core point if within eps, otherwise noise.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Cluster labels for each sample. Noisy samples are given the label -1.
        """
        if self.labels_ is None:
            raise RuntimeError("DBSCAN model has not been fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=np.float64)
        X_fit = self.components_
        labels_fit = self.labels_[self.core_sample_indices_]
        new_labels = np.full(X.shape[0], -1, dtype=int)

        for i, x in enumerate(X):
            dists = np.linalg.norm(X_fit - x, axis=1)
            mask = dists <= self.eps
            if np.any(mask):
                new_labels[i] = labels_fit[mask][0]  # assign to first matching core point
        return new_labels

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Perform DBSCAN clustering from features, or distance matrix, and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Cluster labels for each sample. Noisy samples are given the label -1.
        """
        self.fit(X, y=y, sample_weight=sample_weight)
        return self.labels_
