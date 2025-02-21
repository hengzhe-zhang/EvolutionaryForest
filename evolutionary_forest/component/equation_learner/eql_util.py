import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from evolutionary_forest.component.equation_learner.utils import pretty_print, functions
from evolutionary_forest.component.equation_learner.utils.regularization import (
    L12Smooth,
)
from evolutionary_forest.component.equation_learner.utils.symbolic_network import (
    SymbolicNet,
)


class SymbolicNetSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = SymbolicNet(*args, **kwargs)
        return cls._instance


def symbolic_regression(
    x,
    y,
    var_names=None,
    n_layers=2,
    reg_weight=5e-3,
    learning_rate=1e-2,
    n_epochs=20000,  # total number of epochs for one-stage training
    summary_step=1000,
    initial_weights_sd=(0.1, 0.5, 1.0),
    validation_split=0.2,
    patience=10,
    min_delta=1e-6,
    device=None,
    verbose=True,
):
    """
    Trains the deep symbolic regression network on the provided dataset (x, y)
    and returns a symbolic expression for the relationship between x and y.

    This version uses a one-stage training process with a cosine annealing learning
    rate scheduler for smooth decay of the learning rate.

    Parameters:
      x: Input data (numpy array or torch tensor) of shape (N, input_dim)
      y: Target data (numpy array or torch tensor) of shape (N, 1) or (N,)
      var_names: Optional list of variable names. If None or insufficient, defaults
                 to ["x1", "x2", ..., "xN"] where N is the input dimensionality.
      n_layers: Number of hidden layers in the network.
      reg_weight: Regularization weight for L₁⁄₂ regularization.
      learning_rate: Base learning rate for training.
      n_epochs: Maximum number of epochs for training.
      summary_step: Frequency (in epochs) at which to evaluate and print training progress.
      initial_weights_sd: Tuple of standard deviations (init_sd_first, init_sd_middle, init_sd_last)
                          for weight initialization.
      validation_split: Fraction of data to use as validation (0 <= validation_split < 1).
      patience: Number of summary evaluations with no improvement on the validation loss
                before early stopping is triggered.
      min_delta: Minimum change in validation loss to qualify as an improvement.
      device: Torch device to use; if None, uses 'cuda:0' if available, else 'cpu'.
      verbose: If True, print log messages; if False, disable all logging.

    Returns:
      expr: A string representing the learned symbolic expression.
    """
    # Convert inputs to torch tensors if needed.
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)

    # Ensure target tensor has shape (N, 1) for consistency with network output.
    if y.dim() == 1:
        y = y.unsqueeze(1)

    # Determine input dimensionality.
    input_dim = x.shape[1]

    # Infer variable names if not provided or if not enough names are given.
    if var_names is None or len(var_names) < input_dim:
        var_names = [f"x{i + 1}" for i in range(input_dim)]

    # Split the data into training and validation sets.
    N = x.shape[0]
    n_val = int(N * validation_split)
    n_val = (
        max(n_val, 1) if validation_split > 0 else 0
    )  # ensure at least one sample if split is used
    n_train = N - n_val
    indices = torch.randperm(N)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_x, train_y = x[train_idx], y[train_idx]
    if n_val > 0:
        val_x, val_y = x[val_idx], y[val_idx]
    else:
        # if no validation, use training data as fallback (not recommended)
        val_x, val_y = train_x, train_y

    # Define the set of activation functions used by the network.
    activation_funcs = (
        [functions.Constant()] * 2
        + [functions.Identity()] * 4
        + [functions.Square()] * 4
        + [functions.Sin()] * 2
        + [functions.Cos()] * 2
        + [functions.Product(norm=1)] * 2
    )
    width = len(activation_funcs)
    n_double = functions.count_double(activation_funcs)

    # Unpack standard deviations and initialize weights.
    init_sd_first, init_sd_middle, init_sd_last = initial_weights_sd
    W1 = torch.fmod(
        torch.normal(0, init_sd_first, size=(input_dim, width + n_double)), 2
    )
    W2 = torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2)
    W3 = torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2)
    W4 = torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
    initial_weights = [W1, W2, W3, W4]

    # Instantiate the symbolic network.
    net = SymbolicNetSingleton(
        n_layers, funcs=activation_funcs, initial_weights=initial_weights
    ).to(device)

    # Set up loss function, optimizer, and learning rate scheduler.
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5
    )

    t0 = time.time()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    # Single-stage training loop with cosine annealing.
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = net(train_x)
        regularization = L12Smooth()
        mse_loss = criterion(outputs, train_y)
        reg_loss = regularization(net.get_weights_tensor())
        loss = mse_loss + reg_weight * reg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % summary_step == 0:
            with torch.no_grad():
                val_outputs = net(val_x)
                val_loss = criterion(val_outputs, val_y).item()
            if verbose:
                print(
                    "Epoch: %d\tTrain Loss: %f\tValidation Loss: %f"
                    % (epoch, loss.item(), val_loss)
                )
        if best_val_loss - val_loss > min_delta:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            if verbose:
                print("Early stopping triggered at epoch", epoch)
            break

    t1 = time.time()
    if verbose:
        print("Training completed in %f seconds." % (t1 - t0))

    # Retrieve learned weights and convert them into a symbolic expression.
    with torch.no_grad():
        weights = net.get_weights()
        expr = pretty_print.network(weights, activation_funcs, var_names[:input_dim])
        if verbose:
            print("Learned expression:", expr)

    return expr


if __name__ == "__main__":
    x_data = np.random.uniform(-5, 5, size=(256, 2))
    y_data = (
        np.sin(x_data[:, 0]) + x_data[:, 1]
    )  # or any function generating your target values
    expr = symbolic_regression(
        x_data, y_data, n_layers=1, summary_step=10, patience=10, verbose=True
    )
    print("Final symbolic expression:", expr)
    expr = symbolic_regression(
        x_data, y_data, n_layers=1, summary_step=10, patience=10, verbose=True
    )
    print("Final symbolic expression:", expr)
    expr = symbolic_regression(
        x_data, y_data, n_layers=1, summary_step=10, patience=10, verbose=True
    )
    print("Final symbolic expression:", expr)
    y_data = (
        np.sin(x_data[:, 0]) + x_data[:, 0] ** 2
    )  # or any function generating your target values
    expr = symbolic_regression(
        x_data, y_data, n_layers=1, summary_step=10, patience=10, verbose=True
    )
    print("Final symbolic expression:", expr)
    expr = symbolic_regression(
        x_data, y_data, n_layers=1, summary_step=10, patience=10, verbose=True
    )
    print("Final symbolic expression:", expr)
    y_data = (
        np.sin(x_data[:, 1]) + x_data[:, 1] ** 2
    )  # or any function generating your target values
    expr = symbolic_regression(
        x_data, y_data, n_layers=1, summary_step=10, patience=10, verbose=True
    )
    print("Final symbolic expression:", expr)
    expr = symbolic_regression(
        x_data, y_data, n_layers=1, summary_step=10, patience=10, verbose=True
    )
    print("Final symbolic expression:", expr)
