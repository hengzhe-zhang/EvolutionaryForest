import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from evolutionary_forest.model.clustering.gap_statistic import gap_statistic
from evolutionary_forest.model.cosine_kmeans import CosineKMeans


def determine_optimal_k(X, k_values, method="silhouette"):
    scores = []
    if method == "gap":
        # Use the gap_statistic function to determine optimal_k for this dataset
        optimal_k = gap_statistic(X, k_values)
        return optimal_k

    for k in k_values:
        if k >= len(X):
            continue
        # Initialize the KMeans model
        kmeans = CosineKMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(X)

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X_normalized = X / norms

        if method == "silhouette":
            score = silhouette_score(X_normalized, labels)
        elif method == "davies":
            score = davies_bouldin_score(X_normalized, labels)
        elif method == "calinski":
            score = calinski_harabasz_score(X_normalized, labels)
        else:
            raise ValueError(
                "Invalid method. Choose from 'silhouette', 'davies-bouldin', 'calinski-harabasz', or 'gap'."
            )

        scores.append(score)

    # For Silhouette and Calinski-Harabasz, higher is better, so we find max score
    # For Davies-Bouldin, lower is better, so we find min score
    if method in ["silhouette", "calinski-harabasz"]:
        optimal_k = k_values[np.argmax(scores)]
    elif method == "davies":
        optimal_k = k_values[np.argmin(scores)]
    return optimal_k
