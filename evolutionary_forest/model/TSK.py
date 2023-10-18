import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from torch import optim


class TSKFuzzySystem(BaseEstimator, RegressorMixin):
    def __init__(self, n_rules=3, learning_rate=0.01, batch_size=32, n_epochs=100, drop_rate=0.5, verbose=False):
        self.n_rules = n_rules
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.drop_rate = drop_rate
        self.verbose = verbose
        # These will be PyTorch parameters later
        self.centers_ = None
        self.sigmas_ = None
        self.regressors_ = []

    def fit(self, X, y):
        # Convert X and y to PyTorch tensors
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        # 1. Use clustering for initial centers and then convert to PyTorch Parameters
        kmeans = KMeans(n_clusters=self.n_rules).fit(X.numpy())
        self.centers_ = torch.nn.Parameter(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        self.sigmas_ = torch.nn.Parameter(
            torch.tensor(np.std(X.numpy() - kmeans.transform(X.numpy()).argmin(axis=1, keepdims=True), axis=0),
                         dtype=torch.float32))

        # 2. For each rule, initialize a linear regressor with Ridge Regression
        X_numpy = X.numpy()
        y_numpy = y.numpy()
        for r in range(self.n_rules):
            # Calculate firing levels for all data points
            firing_levels = self._compute_firing_levels(X, r).detach().numpy()

            # Ridge Regression
            ridge = Ridge(alpha=1.0)  # You can adjust the regularization strength with the alpha parameter
            ridge.fit(X_numpy, y_numpy, sample_weight=firing_levels)

            # Bias term + Weights
            bias_and_weights = np.concatenate(([ridge.intercept_], ridge.coef_), axis=0)

            # Convert to PyTorch parameter
            weight = torch.nn.Parameter(torch.tensor(bias_and_weights, dtype=torch.float32))
            self.regressors_.append(weight)

        # Optimizer (using SGD as an example)
        optimizer = optim.Adam([self.centers_, self.sigmas_] + self.regressors_, lr=self.learning_rate)

        # Iterative training using MBGD
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0  # to compute average loss per epoch
            permutation = torch.randperm(X.size()[0])
            num_batches = X.size()[0] // self.batch_size

            for i in range(0, X.size()[0], self.batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = X[indices], y[indices]

                predictions = self._forward(batch_x, training=True)
                loss = torch.nn.functional.mse_loss(predictions, batch_y)
                epoch_loss += loss.item()

                loss.backward()
                optimizer.step()

            epoch_loss /= num_batches  # average loss

            if self.verbose:  # print training log if verbose is True
                print(f"Epoch {epoch + 1}/{self.n_epochs} - Loss: {epoch_loss:.4f}")

        return self

    def _forward(self, X, training=False):
        # Add bias term to X

        rule_outputs = []
        assert len(self.regressors_) == self.n_rules, f"{len(self.regressors_)} != {self.n_rules}"
        for r, weight in enumerate(self.regressors_):
            firing_level = self._compute_firing_levels(X, r)

            # Apply DropRule during training
            if training and self.drop_rate > 0:
                drop_mask = torch.bernoulli(torch.tensor(self.drop_rate)).type(torch.float32)
                firing_level = firing_level * drop_mask

            X_bias = torch.cat((torch.ones(X.shape[0], 1, dtype=torch.float32), X), dim=1)
            rule_output = firing_level * torch.matmul(X_bias, weight)
            rule_outputs.append(rule_output)

        system_output = sum(rule_outputs) / sum(self._compute_firing_levels(X, r) for r in range(self.n_rules))
        return system_output

    def _compute_firing_levels(self, X, rule_idx):
        return torch.prod(torch.exp(-torch.pow(X - self.centers_[rule_idx], 2)
                                    / (2 * torch.pow(self.sigmas_, 2))), axis=1)

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self._forward(X).numpy()


if __name__ == '__main__':
    from sklearn.datasets import make_regression, load_diabetes
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Generate some synthetic regression data
    # X, y = make_regression(n_samples=1000, n_features=5, noise=0.1)
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_std = scaler_X.fit_transform(X_train)
    X_test_std = scaler_X.transform(X_test)

    y_train_std = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_std = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Train and test the TSK fuzzy system
    tsk = TSKFuzzySystem(n_rules=5, n_epochs=100, learning_rate=0.001, drop_rate=0, verbose=True)
    tsk.fit(X_train_std, y_train_std)
    y_pred_std = tsk.predict(X_test_std)
    # Inverse transform the standardized predictions to get them back to the original scale
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))

    lr = LinearRegression()
    lr.fit(X_train_std, y_train_std)
    y_pred_std = lr.predict(X_test_std)
    # Inverse transform the standardized predictions to get them back to the original scale
    y_pred = scaler_y.inverse_transform(y_pred_std.reshape(-1, 1)).flatten()
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))
