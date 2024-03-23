from numpy.linalg import eig


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
