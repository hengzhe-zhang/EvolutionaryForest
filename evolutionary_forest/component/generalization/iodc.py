from scipy.spatial.distance import pdist
import numpy as np


def create_z(X):
    # Calculate the pairwise Euclidean distances of the input data X
    return pdist(X, 'euclidean')


def create_w(Y):
    # Treat the scalar values in Y as vectors containing one single element and calculate the pairwise Euclidean distances
    return pdist(Y.reshape(-1, 1), 'euclidean')


def calculate_iodc(z, w):
    # Calculating the covariance between z and w
    covariance = np.cov(z, w)[0, 1]

    # Calculating the standard deviations of z and w
    sigma_z = np.std(z)
    sigma_w = np.std(w)

    # Calculating the IODC
    iodc = covariance / (sigma_z * sigma_w)

    return iodc


if __name__ == '__main__':
    X = np.array([[1, 2], [2, 3], [3, 4]])
    Y = np.array([2, 3, 4])  # Scalar values corresponding to the outputs

    z = create_z(X)
    w = create_w(Y)

    iodc = calculate_iodc(z, w)
    print("IODC:", iodc)
