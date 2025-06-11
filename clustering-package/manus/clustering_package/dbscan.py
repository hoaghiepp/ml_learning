import numpy as np
from collections import deque
from clustering_package.base import BaseClusteringAlgorithm

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

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def _get_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def fit(self, X, y=None):
        """
        Perform DBSCAN clustering from features, or distance matrix.

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
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1 for noise, 0 for first cluster, etc.
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:  # Already visited or assigned to a cluster
                continue

            neighbors = self._get_neighbors(X, i)

            if len(neighbors) < self.min_samples:  # Not a core point, mark as noise for now
                self.labels_[i] = -1
                continue

            # Found a core point, start a new cluster
            self.labels_[i] = cluster_id
            queue = deque(neighbors.tolist())
            queue.remove(i) # Remove itself from the queue

            while queue:
                current_point = queue.popleft()
                if self.labels_[current_point] == -1:  # If noise, assign to current cluster
                    self.labels_[current_point] = cluster_id
                elif self.labels_[current_point] != -1: # Already visited and assigned to a cluster
                    continue

                current_neighbors = self._get_neighbors(X, current_point)
                if len(current_neighbors) >= self.min_samples:
                    for neighbor in current_neighbors:
                        if self.labels_[neighbor] == -1: # If noise, add to queue
                            self.labels_[neighbor] = cluster_id
                            queue.append(neighbor)

            cluster_id += 1
        return self

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

        X_fit = self._X_fit  # Store X from fit for distance calculations
        labels_fit = self.labels_

        new_labels = np.full(X.shape[0], -1, dtype=int)

        for i, new_point in enumerate(X):
            min_dist = float('inf')
            assigned_cluster = -1

            for j, fitted_point in enumerate(X_fit):
                if labels_fit[j] != -1:  # Only consider core points or border points
                    dist = np.linalg.norm(new_point - fitted_point)
                    if dist <= self.eps and dist < min_dist:
                        min_dist = dist
                        assigned_cluster = labels_fit[j]
            new_labels[i] = assigned_cluster

        return new_labels

    def fit(self, X, y=None):
        """
        Perform DBSCAN clustering from features, or distance matrix.

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
        self._X_fit = np.asarray(X, dtype=np.float64) # Store X for predict method
        n_samples = self._X_fit.shape[0]
        self.labels_ = np.full(n_samples, -1, dtype=int)  # -1 for noise, 0 for first cluster, etc.
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:  # Already visited or assigned to a cluster
                continue

            neighbors = self._get_neighbors(self._X_fit, i)

            if len(neighbors) < self.min_samples:  # Not a core point, mark as noise for now
                self.labels_[i] = -1
                continue

            # Found a core point, start a new cluster
            self.labels_[i] = cluster_id
            queue = deque(neighbors.tolist())
            # Remove already processed points from the queue
            for processed_idx in [idx for idx in queue if self.labels_[idx] != -1]:
                queue.remove(processed_idx)

            while queue:
                current_point_idx = queue.popleft()
                if self.labels_[current_point_idx] != -1: # Already visited and assigned to a cluster
                    continue

                self.labels_[current_point_idx] = cluster_id
                current_neighbors = self._get_neighbors(self._X_fit, current_point_idx)

                if len(current_neighbors) >= self.min_samples:
                    for neighbor_idx in current_neighbors:
                        if self.labels_[neighbor_idx] == -1: # If noise, add to queue
                            queue.append(neighbor_idx)

            cluster_id += 1
        return self


