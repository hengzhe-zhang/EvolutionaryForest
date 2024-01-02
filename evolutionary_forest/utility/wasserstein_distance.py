import torch


# Calculate the covariance matrices
def covariance_matrix(data, mean_data):
    n_samples = data.shape[0]
    data_centered = data - mean_data  # Subtract the mean for each feature
    cov_matrix = torch.mm(data_centered.T, data_centered) / (n_samples - 1)
    return cov_matrix


def safe_sqrt(x, eps=1e-8):
    return torch.sqrt(torch.clamp(x, min=eps))


def wasserstein_distance_torch(mu1, mu2, Sigma1, Sigma2, eps=1e-8):
    mean_diff = mu1 - mu2
    mean_diff_sq = torch.dot(mean_diff, mean_diff)

    # Ensure Sigma1 is symmetric positive definite
    eigvals, eigvecs = torch.linalg.eigh(Sigma1)
    # Addressing potential negative or near-zero eigenvalues
    eigvals = safe_sqrt(eigvals, eps)

    sqrt_Sigma1 = eigvecs @ torch.diag(eigvals) @ eigvecs.t()

    product = sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1
    eigvals_prod, eigvecs_prod = torch.linalg.eigh(product)
    # Again addressing potential negative or near-zero eigenvalues
    eigvals_prod = safe_sqrt(eigvals_prod, eps)

    sqrt_product = eigvecs_prod @ torch.diag(eigvals_prod) @ eigvecs_prod.t()

    w2_distance = (
        mean_diff_sq
        + torch.trace(Sigma1)
        + torch.trace(Sigma2)
        - 2 * torch.trace(sqrt_product)
    )

    return w2_distance


if __name__ == "__main__":
    # Example usage:
    # Define mean and covariance for two Gaussian distributions
    mu1 = torch.tensor([0.0, 0.0])
    mu2 = torch.tensor([1.0, 1.0])
    Sigma1 = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    Sigma2 = torch.tensor([[1.2, 0.2], [0.2, 1.2]])

    # Calculate the 2nd Wasserstein distance
    w2_dist = wasserstein_distance_torch(mu1, mu2, Sigma1, Sigma2)
    print("2nd Wasserstein distance:", w2_dist)
