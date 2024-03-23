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
        Computes column means of the features matrix.
        :return: np.ndarray
        """
        assert self.X is not None, 'PCA not fitted to any data yet. Kindly apply fit first.'
        mean = np.sum(self.X, axis=0, keepdims=True) / self.n_samples
        return mean

    def std(self):
        """
        Computes column standard deviations of the features matrix.
        :return: np.ndarray
        """
        assert not self.X is None, 'PCA not fitted to any data yet. Kindly apply fit first.'
        std = np.sqrt(np.sum((self.X - self.mean()) ** 2, keepdims=True, axis=0) / (self.n_samples - 1))
        return std

    def standardize_data(self):
        """
        Standardize features matrix.
        :return: np.ndarray
        """
        assert not self.X is None, 'PCA not fitted to any data yet. Kindly apply fit first.'
        standardized_data = (self.X - self.mean()) / self.std()
        return standardized_data

    def covariance(self):
        """
        Computes covariance matrix of centered features matrix.
        :return: np.ndarray
        """
        assert self.X is not None, 'PCA not fitted to any data yet. Kindly apply fit first.'
        x = self.standardize_data()
        cov = (x.T @ x) / (self.n_samples - 1)
        return cov

    def eig_vals_vecs(self):
        """
        Computes eigenvalues and eigenvectors of covariance matrix.
        :return: tuple
        """
        eig_vals, eig_vecs = eig(self.covariance())

        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        eig_vecs = eig_vecs[:, idx]
        return eig_vals, eig_vecs

    def explained_variance(self):
        """
        Computes proportion of variance explained by eigenvalues.
        :return: tuple
        """
        eig_vals, _ = self.eig_vals_vecs()
        proportions = [100 * eig_val / sum(eig_vals) for eig_val in eig_vals]
        cum_proportions = np.cumsum(proportions)
        return proportions, cum_proportions

    def principal_components(self):
        """
        computes principal components by selecting the highest n_components eigenvalues.
        :return: np.ndarray
        """
        eig_vals, eig_vecs = self.eig_vals_vecs()
        principal_vals = eig_vals[:self.n_components]
        principal_components = eig_vecs[:, :self.n_components]
        return principal_components

    def transform(self):
        """
        computes projected data onto principal compoents.
        :return: np.ndarray
        """
        p_comps = self.principal_components()
        stand_data = self.standardize_data()
        return stand_data @ p_comps

