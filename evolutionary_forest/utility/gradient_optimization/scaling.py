import torch
from sklearn.preprocessing import StandardScaler


def feature_standardization_torch(constructed_features):
    # normalize before ridge regression
    mean = constructed_features.mean(dim=0)
    std = constructed_features.std(dim=0)
    # mean = constructed_features.mean(dim=0).detach()
    # std = constructed_features.var(dim=0).detach()
    epsilon = 1e-10
    constructed_features_normalized = (constructed_features - mean) / (std + epsilon)
    return constructed_features_normalized


def gradient_agnostic_standarization(features: torch.Tensor, scaler: StandardScaler):
    mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    std = torch.tensor(scaler.scale_, dtype=torch.float32)
    # Check if std is not zero for each feature
    non_zero_std_indices = std != 0
    # Normalize features for non-zero std features
    if non_zero_std_indices.any():
        features[:, non_zero_std_indices] = (
            features[:, non_zero_std_indices] - mean[non_zero_std_indices]
        ) / std[non_zero_std_indices]
    # Subtract mean for zero std features
    features[:, ~non_zero_std_indices] -= mean[~non_zero_std_indices]
