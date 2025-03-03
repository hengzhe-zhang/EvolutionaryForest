import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from evolutionary_forest.component.equation_learner.utils import pretty_print, functions
from evolutionary_forest.component.equation_learner.utils.regularization import (
    L12Smooth,
)
from evolutionary_forest.component.equation_learner.utils.symbolic_network import (
    SymbolicNet,
)

DEBUG = False


class EQLSymbolicRegression:
    def __init__(
        self, n_layers=2, var_names=None, learning_rate=1e-2, device=None, verbose=True
    ):
        """
        Initialize the Symbolic Regression model.

        Parameters:
          n_layers: Number of hidden layers in the network.
          var_names: Optional list of variable names. If None, inferred during fit.
          learning_rate: Base learning rate for training.
          device: Torch device to use; if None, uses 'cuda:0' if available, else 'cpu'.
          verbose: If True, print log messages; if False, disable all logging.
        """
        self.n_layers = n_layers
        self.var_names = var_names
        self.learning_rate = learning_rate
        self.verbose = verbose

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.net = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.initialized = False
        self.input_dim = None
        self.best_val_loss = float("inf")
        self.learned_expr = None

    def _initialize_network(self, input_dim):
        """Initialize the symbolic network with the appropriate dimensions."""
        self.input_dim = input_dim

        # Define the set of activation functions used by the network.
        self.activation_funcs = (
            [functions.Constant()] * 2
            + [functions.Identity()] * 4
            + [functions.Square()] * 4
            + [functions.Sin()] * 2
            + [functions.Cos()] * 2
            + [functions.Product(norm=1)] * 2
        )
        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)

        # Initialize weights.
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
        self.net = SymbolicNet(
            self.n_layers, funcs=self.activation_funcs, initial_weights=initial_weights
        ).to(self.device)

        # Set up optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        # Cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=1000, T_mult=1, eta_min=1e-5
        )

        self.initialized = True

    def fit(
        self,
        x,
        y,
        n_epochs=20000,
        batch_size=32,
        reg_weight=5e-3,
        summary_step=1,
        validation_split=0.2,
        patience=20,
        min_delta=1e-6,
        continue_training=False,
    ):
        """
        Trains the deep symbolic regression network on the provided dataset (x, y).

        Parameters:
          x: Input data (numpy array or torch tensor) of shape (N, input_dim)
          y: Target data (numpy array or torch tensor) of shape (N, 1) or (N,)
          n_epochs: Maximum number of epochs for training.
          batch_size: Batch size for mini-batch training.
          reg_weight: Regularization weight for L₁⁄₂ regularization.
          summary_step: Frequency (in epochs) at which to evaluate and print training progress.
          validation_split: Fraction of data to use as validation (0 <= validation_split < 1).
          patience: Number of summary evaluations with no improvement on the validation loss
                    before early stopping is triggered.
          min_delta: Minimum change in validation loss to qualify as an improvement.
          continue_training: If True and model is already initialized, continue training
                            from the current weights. If False, reinitialize the model.

        Returns:
          self: The fitted model instance.
        """
        # Convert inputs to torch tensors if needed.
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32)

        x = x.to(self.device)
        y = y.to(self.device)

        # Ensure target tensor has shape (N, 1) for consistency with network output.
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Determine input dimensionality.
        input_dim = x.shape[1]

        # Initialize model if not already initialized or if continuing training
        # with different dimensions or if not continuing training
        if (
            not self.initialized
            or (self.input_dim != input_dim)
            or not continue_training
        ):
            # Infer variable names if not provided or if not enough names are given.
            if self.var_names is None or len(self.var_names) < input_dim:
                self.var_names = [f"x{i + 1}" for i in range(input_dim)]

            # Initialize the network
            self._initialize_network(input_dim)
            self.best_val_loss = float("inf")

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

        t0 = time.time()

        # Training loop
        patience_counter = 0

        # Single-stage training loop with cosine annealing.
        for epoch in range(n_epochs):
            self.net.train()
            epoch_loss = 0.0
            n_batches = len(train_loader)
            for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(batch_x)
                regularization = L12Smooth()
                mse_loss = self.criterion(outputs, batch_y)
                reg_loss = regularization(self.net.get_weights_tensor())
                loss = mse_loss + reg_weight * reg_loss
                loss.backward()
                self.optimizer.step()

                # Update scheduler with fractional epoch index.
                self.scheduler.step(epoch + batch_idx / n_batches)

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / n_batches
            if epoch % summary_step == 0:
                # Evaluate on validation set at summary intervals.
                self.net.eval()
                with torch.no_grad():
                    val_outputs = self.net(val_x)
                    val_loss = self.criterion(val_outputs, val_y).item()
                if self.verbose:
                    print(
                        f"Epoch: {epoch}\tAvg Train Loss: {avg_epoch_loss:.6f}\tValidation Loss: {val_loss:.6f}"
                    )

                if self.best_val_loss - val_loss > min_delta:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
            if patience_counter >= patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}")
                break

        t1 = time.time()
        if self.verbose:
            print(f"Training completed in {t1 - t0:.2f} seconds.")

        # Update the learned expression
        self._update_expression()

        return self

    def _update_expression(self):
        """Update the learned symbolic expression from the current network weights."""
        self.net.eval()
        with torch.no_grad():
            weights = self.net.get_weights()
            self.learned_expr = pretty_print.network(
                weights, self.activation_funcs, self.var_names[: self.input_dim]
            )
            if self.verbose:
                print("Learned expression:", self.learned_expr)

    def predict(self, x):
        """
        Make predictions using the trained network.

        Parameters:
          x: Input data (numpy array or torch tensor) of shape (N, input_dim)

        Returns:
          y_pred: Predicted values as numpy array of shape (N, 1)
        """
        if not self.initialized:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        # Convert inputs to torch tensors if needed.
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        self.net.eval()
        with torch.no_grad():
            y_pred = self.net(x)

        return y_pred.cpu().numpy()

    def get_expression(self):
        """Return the learned symbolic expression."""
        if not self.initialized:
            raise ValueError("Model has not been trained yet. Call fit() first.")
        return self.learned_expr

    def save_model(self, path):
        """
        Save the model to a file.

        Parameters:
          path: File path to save the model
        """
        if not self.initialized:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        state = {
            "n_layers": self.n_layers,
            "var_names": self.var_names,
            "learning_rate": self.learning_rate,
            "input_dim": self.input_dim,
            "best_val_loss": self.best_val_loss,
            "learned_expr": self.learned_expr,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        torch.save(state, path)

    def load_model(self, path):
        """
        Load a saved model.

        Parameters:
          path: File path to load the model from

        Returns:
          self: The loaded model instance
        """
        state = torch.load(path, map_location=self.device)

        self.n_layers = state["n_layers"]
        self.var_names = state["var_names"]
        self.learning_rate = state["learning_rate"]
        self.input_dim = state["input_dim"]
        self.best_val_loss = state["best_val_loss"]
        self.learned_expr = state["learned_expr"]

        # Initialize network with correct dimensions
        self._initialize_network(self.input_dim)

        # Load saved states
        self.net.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])

        return self


if __name__ == "__main__":
    # Create a synthetic dataset
    np.random.seed(0)
    x = np.random.rand(1000, 2) * 10  # 1000 samples, 2 features
    y = (
        x[:, 0] ** 2 + np.sin(x[:, 1]) + 0.1 * np.random.randn(1000)
    )  # Target function with noise

    # Initialize the model
    model = EQLSymbolicRegression(n_layers=1, var_names=["x", "y"], verbose=True)

    # First training round
    print("First training round:")
    model.fit(
        x, y, n_epochs=5000, batch_size=32, reg_weight=5e-3, continue_training=True
    )

    # Second training round
    print("\nSecond training round (continuing):")
    model.fit(
        x, y, n_epochs=5000, batch_size=32, reg_weight=1e-2, continue_training=True
    )

    # Get the final expression
    expr = model.get_expression()
    print(f"\nFinal learned expression: {expr}")
