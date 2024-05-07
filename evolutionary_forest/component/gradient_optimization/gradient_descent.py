import numpy as np
import torch
import torch.optim as optim
from deap.gp import Terminal
from sklearn.linear_model import Ridge

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
