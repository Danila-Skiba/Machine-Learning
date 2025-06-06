import numpy as np


class PCACustom:
    def __init__(self, n_components = 2):
        self.n_components = n_components
        self.components = None
        self.eigenvalues = None
        self.mean= None

    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X_center = X - self.mean

        cov_mat = np.cov(X_center.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)

        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[idx]
        self.components = eigenvectors[:, idx][:, :self.n_components]

        return self
    
    def transform(self, X):
        X_center = X - self.mean
        X_transformed = np.dot(X_center, self.components)
        return X_transformed
    
    def fit_transform(self,X):
        return self.fit(X).transform(X)