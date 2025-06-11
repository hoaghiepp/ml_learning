from abc import ABC, abstractmethod

class BaseClusteringAlgorithm(ABC):
    """
    Abstract base class for all clustering algorithms.
    Defines the common interface for fit and predict methods.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the clustering model to the training data.

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
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the cluster labels for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        labels : array-like of shape (n_samples,)
            Cluster labels for each sample.
        """
        pass


