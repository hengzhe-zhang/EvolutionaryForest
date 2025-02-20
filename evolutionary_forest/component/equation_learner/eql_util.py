import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from inspect import signature

# Import required modules from your utils package.
from utils import pretty_print, functions
from utils.symbolic_network import SymbolicNet
from utils.regularization import L12Smooth


def symbolic_regression(x, y,
                        var_names=["x", "y", "z"],
                        n_layers=2,
                        reg_weight=5e-3,
                        learning_rate=1e-2,
                        n_epochs1=10001,
                        n_epochs2=10001,
                        summary_step=1000,
                        initial_weights_sd=(0.1, 0.5, 1.0),
                        device=None):
    """
    Trains the deep symbolic regression architecture on the provided dataset (x, y)
    and returns a symbolic expression that describes the relationship between x and y.

    Parameters:
      x: Input data (numpy array or torch tensor) of shape (N, input_dim)
      y: Target data (numpy array or torch tensor) of shape (N, 1)
      var_names: List of variable names for the expression (only first input_dim used)
      n_layers: Number of hidden layers in the network.
      reg_weight: Regularization weight for L₁⁄₂ regularization.
      learning_rate: Base learning rate for training.
      n_epochs1: Number of epochs for the first stage of training.
      n_epochs2: Number of epochs for the second stage of training.
      summary_step: Frequency (in epochs) to print training progress.
      initial_weights_sd: Tuple of standard deviations (init_sd_first, init_sd_middle, init_sd_last)
                          for weight initialization.
      device: Torch device to use; if None, uses 'cuda:0' if available, else 'cpu'.

    Returns:
      expr: A string representing the symbolic expression learned from the data.
    """
    # Convert x and y to torch tensors (if not already) and move them to the chosen device.
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = x.to(device)
    y = y.to(device)

    # Determine input dimensionality from provided data.
    input_dim = x.shape[1]

    # Define the set of activation functions used by the EQL network.
    activation_funcs = (
        [functions.Constant()] * 2 +
        [functions.Identity()] * 4 +
        [functions.Square()] * 4 +
        [functions.Sin()] * 2 +
        [functions.Exp()] * 2 +
        [functions.Sigmoid()] * 2 +
        [functions.Product()] * 2
    )

    width = len(activation_funcs)
    n_double = functions.count_double(activation_funcs)

    # Unpack standard deviations for weight initialization.
    init_sd_first, init_sd_middle, init_sd_last = initial_weights_sd

    # Create the list of initial weight tensors.
    W1 = torch.fmod(torch.normal(0, init_sd_first, size=(input_dim, width + n_double)), 2)
    W2 = torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2)
    W3 = torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2)
    W4 = torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
    initial_weights = [W1, W2, W3, W4]

    # Instantiate the symbolic network.
    net = SymbolicNet(n_layers, funcs=activation_funcs, initial_weights=initial_weights).to(device)

    # Set up loss function, optimizer and learning rate scheduler.
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(),
                              lr=learning_rate * 10,
                              alpha=0.9,
                              eps=1e-10,
                              momentum=0.0,
                              centered=False)

    lr_lambda = lambda epoch: 0.1
    scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)

    t0 = time.time()
    loss_val = np.nan
    # If training diverges (loss becomes NaN or too high), restart the training.
    while np.isnan(loss_val):
        # First stage of training (with an additional 2000 epoch warmup)
        for epoch in range(n_epochs1 + 2000):
            optimizer.zero_grad()
            outputs = net(x)
            regularization = L12Smooth()
            mse_loss = criterion(outputs, y)
            reg_loss = regularization(net.get_weights_tensor())
            loss = mse_loss + reg_weight * reg_loss
            loss.backward()
            optimizer.step()

            if epoch % summary_step == 0:
                loss_val = loss.item()
                print("Stage 1 Epoch: %d\tLoss: %f" % (epoch, loss_val))
                if np.isnan(loss_val) or loss_val > 1000:
                    break
            if epoch == 2000:
                scheduler.step()  # Reduce learning rate.

        scheduler.step()  # Further reduce learning rate.
        # Second stage of training.
        for epoch in range(n_epochs2):
            optimizer.zero_grad()
            outputs = net(x)
            regularization = L12Smooth()
            mse_loss = criterion(outputs, y)
            reg_loss = regularization(net.get_weights_tensor())
            loss = mse_loss + reg_weight * reg_loss
            loss.backward()
            optimizer.step()

            if epoch % summary_step == 0:
                loss_val = loss.item()
                print("Stage 2 Epoch: %d\tLoss: %f" % (epoch, loss_val))
                if np.isnan(loss_val) or loss_val > 1000:
                    break

        if np.isnan(loss_val) or loss_val > 1000:
            # Reinitialize the network and optimizer if divergence occurred.
            net = SymbolicNet(n_layers, funcs=activation_funcs, initial_weights=initial_weights).to(device)
            optimizer = optim.RMSprop(net.parameters(),
                                      lr=learning_rate * 10,
                                      alpha=0.9,
                                      eps=1e-10,
                                      momentum=0.0,
                                      centered=False)
            scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
            loss_val = np.nan
        else:
            break  # Training was successful.

    t1 = time.time()
    print("Training completed in %f seconds." % (t1 - t0))

    # Retrieve the learned weights and convert them into a symbolic expression.
    with torch.no_grad():
        weights = net.get_weights()
        # Use only the first 'input_dim' variable names.
        expr = pretty_print.network(weights, activation_funcs, var_names[:input_dim])
        print("Learned expression:", expr)

    return expr


if __name__ == '__main__':
    x_data = np.random.uniform(-1, 1, size=(256, 1))
    y_data = np.sin(x_data)  # or any function generating your target values
    expr = symbolic_regression(x_data, y_data, var_names=["x"])
    print("Final symbolic expression:", expr)
