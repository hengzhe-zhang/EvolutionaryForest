import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from numba import njit


@njit
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return (
            0.0  # If either vector is zero, cosine similarity is not defined; return 0
        )
    return dot_product / (norm_a * norm_b)


@njit
def compute_cosine_similarity_matrix(X, centers):
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    similarity_matrix = np.zeros((n_samples, n_clusters))

    for i in range(n_samples):
        for j in range(n_clusters):
            similarity_matrix[i, j] = cosine_similarity(X[i], centers[j])

    return similarity_matrix


@njit
def update_centers(X, labels, n_clusters):
    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features))
    counts = np.zeros(n_clusters)

    for i in range(X.shape[0]):
        centers[labels[i]] += X[i]
        counts[labels[i]] += 1

    for j in range(n_clusters):
        if counts[j] > 0:
            centers[j] /= counts[j]

    return centers


class CosineKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Initialize centers
        initial_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centers = X[initial_indices]

        for iteration in range(self.max_iter):
            similarity_matrix = compute_cosine_similarity_matrix(X, centers)
            labels = np.argmax(similarity_matrix, axis=1)

            new_centers = update_centers(X, labels, self.n_clusters)
            center_shift = np.sum((centers - new_centers) ** 2)

            centers = new_centers

            if center_shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = labels
        return self

    def predict(self, X):
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X.astype(np.float64))

        similarity_matrix = compute_cosine_similarity_matrix(X, self.cluster_centers_)
        return np.argmax(similarity_matrix, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X.astype(np.float64))
        return self.labels_


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=300, centers=5, random_state=0)

    model = CosineKMeans(n_clusters=5, random_state=0)
    labels = model.fit_predict(X.astype(np.float64))
    print(labels)
