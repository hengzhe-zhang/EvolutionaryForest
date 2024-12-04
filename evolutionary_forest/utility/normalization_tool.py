import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from evolutionary_forest.model.cosine_kmeans import cosine_similarity


def normalize_vector(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().numpy()
    v = v - np.mean(v)
    norm = np.linalg.norm(v)
    if norm == 0:
        # undefined correlation
        return v
    normalized_v = v / norm
    return normalized_v


if __name__ == "__main__":
    # Example usage
    residual_1 = np.array([0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    residual_2 = np.array([-0.1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])

    normalized_residual_1 = normalize_vector(residual_1)
    normalized_residual_2 = normalize_vector(residual_2)

    print(normalized_residual_1)
    print(normalized_residual_2)

    print(cosine_similarity(normalized_residual_1, normalized_residual_2))
    print(pearsonr(normalized_residual_1, normalized_residual_2)[0])

    regression = LinearRegression()
    regression.fit(normalized_residual_1.reshape(-1, 1), normalized_residual_2)
    print(regression.predict(normalized_residual_1.reshape(-1, 1)))
    print(
        "R2",
        r2_score(
            normalized_residual_2,
            regression.predict(normalized_residual_1.reshape(-1, 1)),
        ),
    )
    print(r2_score(normalized_residual_2, -1 * normalized_residual_1))
