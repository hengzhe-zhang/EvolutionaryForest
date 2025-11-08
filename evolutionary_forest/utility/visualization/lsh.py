import numpy as np


def numpy_to_lsh(X: np.ndarray, n_bits: int = 16, random_state: int = 0) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    random_vectors = rng.standard_normal((X.shape[1], n_bits))
    projections = np.dot(X, random_vectors)
    hashes = (projections >= 0).astype(int)
    return hashes.squeeze()
