from abc import ABC, abstractmethod

class IClustering(ABC):
    @abstractmethod
    def fit(self, X, y, sample_weight):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def fit_predict(self, X, y):
        pass