import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import preorder_traversal
from torch.utils.data import TensorDataset, DataLoader

from evolutionary_forest.component.equation_learner.utils import pretty_print, functions
from evolutionary_forest.component.equation_learner.utils.regularization import (
    L12Smooth,
)
from evolutionary_forest.component.equation_learner.utils.symbolic_network import (
    SymbolicNet,
)

DEBUG = False


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
    batch_size=32,  # new parameter for batch training
    summary_step=1,
    validation_split=0.2,
    patience=20,
    min_delta=1e-6,
    device=None,
    verbose=True,
):
    """
    Trains the deep symbolic regression network on the provided dataset (x, y)
    and returns a symbolic expression for the relationship between x and y.

    This version uses mini-batch training with a cosine annealing warm-restart
    learning rate scheduler.

    Parameters:
      x: Input data (numpy array or torch tensor) of shape (N, input_dim)
      y: Target data (numpy array or torch tensor) of shape (N, 1) or (N,)
      var_names: Optional list of variable names. If None or insufficient, defaults
                 to ["x1", "x2", ..., "xN"] where N is the input dimensionality.
      n_layers: Number of hidden layers in the network.
      reg_weight: Regularization weight for L₁⁄₂ regularization.
      learning_rate: Base learning rate for training.
      n_epochs: Maximum number of epochs for training.
      batch_size: Batch size for mini-batch training.
      summary_step: Frequency (in epochs) at which to evaluate and print training progress.
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

    # Create DataLoader for mini-batch training.
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

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
    W1 = torch.empty(size=(input_dim, width + n_double))
    nn.init.xavier_normal_(W1)

    W2 = torch.empty(size=(width, width + n_double))
    nn.init.xavier_normal_(W2)

    W3 = torch.empty(size=(width, width + n_double))
    nn.init.xavier_normal_(W3)

    W4 = torch.empty(size=(width, 1))
    nn.init.xavier_normal_(W4)
    initial_weights = [W1, W2, W3, W4]

    # Instantiate the symbolic network.
    net = SymbolicNetSingleton(
        n_layers, funcs=activation_funcs, initial_weights=initial_weights
    ).to(device)

    # Set up loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        net.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=1000, T_mult=1, eta_min=1e-5
    )

    t0 = time.time()

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    # Single-stage training loop with cosine annealing.
    for epoch in range(n_epochs):
        net.train()
        epoch_loss = 0.0
        n_batches = len(train_loader)
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = net(batch_x)
            regularization = L12Smooth()
            mse_loss = criterion(outputs, batch_y)
            reg_loss = regularization(net.get_weights_tensor())
            loss = mse_loss + reg_weight * reg_loss
            loss.backward()
            optimizer.step()

            # Update scheduler with fractional epoch index.
            scheduler.step(epoch + batch_idx / n_batches)

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / n_batches
        if epoch % summary_step == 0:
            # Evaluate on validation set at summary intervals.
            net.eval()
            with torch.no_grad():
                val_outputs = net(val_x)
                val_loss = criterion(val_outputs, val_y).item()
            if verbose:
                print(
                    "Epoch: %d\tAvg Train Loss: %f\tValidation Loss: %f"
                    % (epoch, avg_epoch_loss, val_loss)
                )

            if best_val_loss - val_loss > min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
        if patience_counter >= patience:
            if verbose or DEBUG:
                print("Early stopping triggered at epoch", epoch)
            break

    t1 = time.time()
    if verbose or DEBUG:
        print("Training completed in %f seconds." % (t1 - t0))

    # Retrieve learned weights and convert them into a symbolic expression.
    net.eval()
    with torch.no_grad():
        weights = net.get_weights()
        expr = pretty_print.network(weights, activation_funcs, var_names[:input_dim])
        if verbose:
            print("Learned expression:", expr)

    return expr


if __name__ == "__main__":
    x_data = np.random.uniform(-5, 5, size=(100, 2))
    y_data = np.sin(x_data[:, 0]) + 2 * x_data[:, 1]

    reg_weight = 5e-3
    summary_step = 1
    patience = 20
    learning_rate = 1e-2
    n_layers = 2

    num_nodes = math.inf
    start = time.time()
    while num_nodes > 30:
        print("Current reg_weight:", reg_weight)
        expr = symbolic_regression(
            x_data,
            y_data,
            n_layers=n_layers,
            learning_rate=learning_rate,
            reg_weight=reg_weight,
            summary_step=summary_step,
            patience=patience,
            verbose=False,
        )
        num_nodes = sum(1 for _ in preorder_traversal(expr))
        reg_weight *= 2
        print("Final symbolic expression:", expr)
    end = time.time()
    print("Time taken:", end - start)
