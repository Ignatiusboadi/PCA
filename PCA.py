from numpy.linalg import eig
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.X = None
        self.n_samples = None
        self.n_features = None

    def fit(self, X):
        """
        fit the PCA class to a features matrix.
        :param X: np.ndarray
            features matrix
        :return: None
        """
        self.X = X
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1]

    def mean(self):
        """
        Computes mean of the features matrix fit.
        :return: None
        """
        assert self.X is not None, 'PCA not fitted to any data yet. Kindly apply fit first.'
        mean = np.sum(self.X, axis=0, keepdims=True) / self.n_samples
        return mean
