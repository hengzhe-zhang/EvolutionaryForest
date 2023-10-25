"""
File: RBFN.py
Author: Octavio Arriaga
Email: arriaga.camargo@email.com
Github: https://github.com/oarriaga
Description: Minimal implementation of a radial basis function network
"""

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.gaussian_process.kernels import RBF


class RBFN(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_shape, sigma=1.0):
        """radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma * np.linalg.norm(center - data_point) ** 2)

    def _calculate_interpolation_matrix(self, X):
        """Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        # G = np.zeros((len(X), self.hidden_shape))
        # for data_point_arg, data_point in enumerate(X):
        #     for center_arg, center in enumerate(self.centers):
        #         G[data_point_arg, center_arg] = self._kernel_function(center, data_point)
        rbf = RBF()
        G = rbf(X, self.centers)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        centers = X[random_args]
        return centers

    def fit(self, X, Y):
        """Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)
        G = self._calculate_interpolation_matrix(X)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions
