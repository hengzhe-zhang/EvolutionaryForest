import numpy as np
from sklearn.metrics import pairwise_distances


def topk_binary_laplacian_fast(y, k=5):
    y = y.reshape(-1, 1)
    dist = pairwise_distances(y)
    idx = np.argsort(dist, axis=1)[:, 1 : k + 1]  # exclude self

    n = len(y)
    S = np.zeros((n, n), dtype=np.float32)

    # vectorized assignment
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    S[rows, cols] = 1.0

    # symmetrize
    S = np.maximum(S, S.T)

    D = np.diag(S.sum(axis=1))
    L = D - S
    return S, L


if __name__ == "__main__":
    y = np.array([0.0, 1.0, 2.0, 5.0])
    S, L = topk_binary_laplacian_fast(y, k=2)
    print(S, L)
