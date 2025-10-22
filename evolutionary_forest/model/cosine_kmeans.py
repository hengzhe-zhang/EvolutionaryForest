import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array


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


class CosineKMeans(KMeans):
    """
    Cosine-similarity KMeans implemented by subclassing sklearn.KMeans.

    This works by L2-normalizing all samples and cluster centers so that
    cosine similarity == dot product. Otherwise identical to regular KMeans.
    """

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, dtype=np.float64)
        X = normalize(X, norm="l2")

        super().fit(X, y, sample_weight)
        self.cluster_centers_ = normalize(self.cluster_centers_, norm="l2")
        return self

    def predict(self, X):
        X = check_array(X, dtype=np.float64)
        X = normalize(X, norm="l2")

        # Cosine similarity = dot product on L2-normalized vectors
        similarity = X @ self.cluster_centers_.T
        return np.argmax(similarity, axis=1)


def select_medoid(cluster_semantics):
    # Compute pairwise cosine similarity within the cluster
    similarity_matrix = compute_cosine_similarity_matrix(
        cluster_semantics, cluster_semantics
    )

    # The medoid is the point with the maximum sum of similarities (equivalent to minimizing distance)
    medoid_idx = np.argmax(np.sum(similarity_matrix, axis=1))

    return medoid_idx


class CosineKMedoids(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X.astype(np.float64))
        n_samples = X.shape[0]
        random_state = check_random_state(self.random_state)

        # Initialize medoids randomly
        medoid_indices = random_state.choice(n_samples, self.n_clusters, replace=False)
        medoids = X[medoid_indices]

        labels = np.zeros(n_samples, dtype=int)

        for iteration in range(self.max_iter):
            # Compute cosine similarity between each point and medoids
            similarity_matrix = compute_cosine_similarity_matrix(X, medoids)
            labels = np.argmax(similarity_matrix, axis=1)

            new_medoids = np.zeros_like(medoids)

            # Update medoids by selecting the most representative point in each cluster
            for cluster_idx in range(self.n_clusters):
                cluster_points = X[labels == cluster_idx]
                if len(cluster_points) == 0:
                    continue  # If no points are assigned to the cluster, skip

                # Compute pairwise cosine similarity within the cluster
                pairwise_similarities = compute_cosine_similarity_matrix(
                    cluster_points, cluster_points
                )

                # Find the point with the maximum sum of similarities (i.e., best medoid)
                medoid_idx = np.argmax(np.sum(pairwise_similarities, axis=1))
                new_medoids[cluster_idx] = cluster_points[medoid_idx]

            # Compute the shift in medoid positions
            medoid_shift = np.sum(np.linalg.norm(medoids - new_medoids, axis=1))

            # Update medoids
            medoids = new_medoids

            # Check for convergence
            if medoid_shift <= self.tol:
                break

        self.cluster_centers_ = medoids
        self.labels_ = labels
        return self

    def predict(self, X):
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X.astype(np.float64))

        # Compute cosine similarity between X and medoids
        similarity_matrix = compute_cosine_similarity_matrix(X, self.cluster_centers_)
        return np.argmax(similarity_matrix, axis=1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    import time

    X, _ = make_blobs(n_samples=3000, centers=5, random_state=0)
    start = time.time()
    model = CosineKMeans(n_clusters=5, random_state=0)
    labels = model.fit_predict(X.astype(np.float64))
    print(labels)
    end = time.time()
    print("Time taken:", end - start)

    start = time.time()
    model = KMeans(n_clusters=5, random_state=0)
    labels = model.fit_predict(X.astype(np.float64))
    print(labels)
    end = time.time()
    print("Time taken:", end - start)
