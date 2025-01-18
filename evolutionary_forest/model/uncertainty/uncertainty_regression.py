import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam


class UncertaintyRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(UncertaintyRegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 16)
        self.mu = nn.Linear(16, 1)  # Output the mean
        self.log_var = nn.Linear(16, 1)  # Output the log variance

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var


class GaussianNLLLoss(nn.Module):
    def forward(self, y_true, mu, log_var):
        var = torch.exp(log_var)
        return torch.mean(0.5 * torch.log(var) + 0.5 * ((y_true - mu) ** 2) / var)


class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_dim,
        lr=0.001,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        patience=10,
        normalize_y=False,
    ):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.normalize_y = normalize_y
        self.model = UncertaintyRegressionModel(input_dim)
        self.loss_fn = GaussianNLLLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler() if normalize_y else None
        self.best_loss = np.inf
        self.early_stop_counter = 0

    def fit(self, X, y):
        # Normalize the input features
        X = self.scaler_X.fit_transform(X)

        # Normalize the target variable if specified
        if self.normalize_y:
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, random_state=42
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        self.model.train()
        for epoch in range(self.epochs):
            # Training loop
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                mu, log_var = self.model(batch_X)
                loss = self.loss_fn(batch_y, mu, log_var)
                loss.backward()
                self.optimizer.step()

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                mu_val, log_var_val = self.model(X_val_tensor)
                val_loss = self.loss_fn(y_val_tensor, mu_val, log_var_val).item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Validation Loss: {val_loss}")

            # Early stopping check
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_state = self.model.state_dict()  # Save the best model
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1

            if self.early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                self.model.load_state_dict(self.best_model_state)  # Load the best model
                break

        return self

    def predict(self, X):
        X = self.scaler_X.transform(X)  # Normalize the input features
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            mu, log_var = self.model(X_tensor)
        mu = mu.numpy()
        var = torch.exp(log_var).numpy()

        # Inverse transform the predictions if target normalization was applied
        if self.normalize_y:
            mu = self.scaler_y.inverse_transform(mu)

        return mu, var


def evaluate_uncertainty(y_true, mu_pred, var_pred):
    std_pred = np.sqrt(var_pred)

    within_1_std = np.mean(np.abs(y_true - mu_pred) <= std_pred)
    within_2_std = np.mean(np.abs(y_true - mu_pred) <= 2 * std_pred)
    within_3_std = np.mean(np.abs(y_true - mu_pred) <= 3 * std_pred)

    print(f"Proportion within 1 std: {within_1_std:.2f}")
    print(f"Proportion within 2 std: {within_2_std:.2f}")
    print(f"Proportion within 3 std: {within_3_std:.2f}")

    return within_1_std, within_2_std, within_3_std


def plot_uncertainty_vs_error(y_true, mu_pred, var_pred):
    std_pred = np.sqrt(var_pred)
    errors = np.abs(y_true - mu_pred)

    # Check the shapes of the arrays
    assert len(std_pred) == len(errors), (
        "Predicted std and errors must have the same length"
    )

    plt.figure(figsize=(10, 6))
    plt.scatter(std_pred, errors, alpha=0.5)
    plt.xlabel("Predicted Standard Deviation")
    plt.ylabel("Absolute Prediction Error")
    plt.title("Uncertainty vs. Prediction Error")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    X, y = load_diabetes(return_X_y=True)

    # Example dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and fit the model
    model = PyTorchRegressor(
        input_dim=X_train.shape[1],
        lr=0.001,
        epochs=1000,
        batch_size=32,
        validation_split=0.2,
        patience=10,
        normalize_y=True,
    )
    model.fit(X_train, y_train)

    # Make predictions
    mu_pred, var_pred = model.predict(X_test)

    # Evaluate the model
    mse = r2_score(y_test, mu_pred)
    print(f"Mean Squared Error: {mse}")

    # Evaluate the predicted uncertainty
    evaluate_uncertainty(y_test, mu_pred, var_pred)

    mu_pred = mu_pred.flatten()
    var_pred = var_pred.flatten()
    y_test = y_test.flatten()
    # Plot uncertainty vs. error
    plot_uncertainty_vs_error(y_test, mu_pred, var_pred)
