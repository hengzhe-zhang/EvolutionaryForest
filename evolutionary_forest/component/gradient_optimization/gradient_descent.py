import numpy as np
import torch
import torch.optim as optim
from deap.gp import Terminal
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

from evolutionary_forest.model.weight_solver import solve_transformation_matrix
from evolutionary_forest.utility.gradient_optimization.scaling import (
    feature_standardization_torch,
)


def fit_ridge_model(features, target):
    ridge = Ridge()
    ridge.fit(features.detach().numpy(), target)
    assert isinstance(ridge, Ridge), "Only linear models are supported here."
    return ridge.coef_, ridge.intercept_


def initialize_torch_variables(weights, bias, optimize_lr_weights):
    requires_grad = optimize_lr_weights
    weights_torch = torch.tensor(
        weights, dtype=torch.float32, requires_grad=requires_grad
    )
    bias_torch = torch.tensor(bias, dtype=torch.float32, requires_grad=requires_grad)
    return weights_torch, bias_torch


def calculate_predictions(features, weights, bias):
    return torch.mm(features, weights.view(-1, 1)) + bias


def perform_gradient_descent(variables, Y_pred, Y, lr=1.0):
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(variables, lr=lr)
    loss = criterion(Y_pred, torch.from_numpy(Y).detach().float())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def contrastive_gradient_optimization(constructed_features, Y, func):
    constructed_features_normalized = feature_standardization_torch(
        constructed_features
    )
    knn_subsampling = 100
    subsample_indices = np.random.choice(len(Y), knn_subsampling, replace=False)
    GP_X_subsample = constructed_features_normalized[subsample_indices]
    y_subsample = Y[subsample_indices]
    D_group = pairwise_distances(y_subsample.reshape(-1, 1), metric="euclidean")

    weight = solve_transformation_matrix(
        GP_X_subsample.detach().numpy(),
        D_group,
    )
    weights_torch = torch.tensor(weight, dtype=torch.float32, requires_grad=True)
    free_variables = [
        f.value
        for tree in func
        for f in tree
        if isinstance(f, Terminal) and isinstance(f.value, torch.Tensor)
    ]

    for v in free_variables:
        assert v.requires_grad is True

    torch_variables = [weights_torch] + free_variables
    # print(free_variables)
    criterion = torch.nn.MSELoss()
    optimizer = optim.SGD(torch_variables, lr=0.1)
    Y_pred = GP_X_subsample.to(dtype=torch.float32) @ weights_torch

    # Step 1: Compute squared norms for each row
    squared_norms = (Y_pred**2).sum(dim=1).unsqueeze(1)  # Shape: (100, 1)

    # Step 2: Compute pairwise squared distances
    pairwise_distances_squared = (
        squared_norms - 2 * Y_pred @ Y_pred.T + squared_norms.T
    )  # Shape: (100, 100)

    loss = criterion(
        pairwise_distances_squared.flatten(),
        torch.from_numpy(D_group.flatten()).detach().float(),
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def gradient_optimization(constructed_features, Y, configuration, func):
    constructed_features_normalized = feature_standardization_torch(
        constructed_features
    )
    weights, bias = fit_ridge_model(constructed_features_normalized, Y)

    if np.all(weights == 0):
        return  # All zero weights mean no optimization is needed

    optimize_lr_weights = True
    weights_torch, bias_torch = initialize_torch_variables(
        weights, bias, optimize_lr_weights
    )

    Y_pred = calculate_predictions(
        constructed_features_normalized, weights_torch, bias_torch
    )

    free_variables = [
        f.value
        for tree in func
        for f in tree
        if isinstance(f, Terminal) and isinstance(f.value, torch.Tensor)
    ]

    if optimize_lr_weights:
        torch_variables = [weights_torch, bias_torch] + free_variables
    else:
        torch_variables = free_variables

    for v in torch_variables:
        assert v.requires_grad is True

    if len(free_variables) >= 1:
        if configuration.constant_type in ["GD", "GD+"]:
            perform_gradient_descent(torch_variables, Y_pred, Y)
        else:
            raise Exception("Unsupported optimization configuration.")
