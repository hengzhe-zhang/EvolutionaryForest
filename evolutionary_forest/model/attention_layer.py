import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping


class AttentionNet(nn.Module):
    def __init__(self, input_dim, model_output_dim, hidden_dim, output_dim):
        super(AttentionNet, self).__init__()
        self.model_output_dim = model_output_dim
        self.attention_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden = nn.Linear(input_dim, model_output_dim)
        self.output_layer = nn.Linear(model_output_dim, output_dim)

    def forward(self, x):
        x = x.float()  # Ensure input is float32
        original_input, model_outputs = (
            x[:, : -self.model_output_dim],
            x[:, -self.model_output_dim :],
        )
        attention_weights = F.softmax(
            self.hidden(self.attention_layer(original_input)), dim=1
        )
        weighted_model_outputs = torch.sum(model_outputs * attention_weights, dim=1)
        return weighted_model_outputs


class SkorchAttentionNet(NeuralNetRegressor):
    def __init__(
        self, *args, input_dim, model_output_dim, hidden_dim, output_dim, **kwargs
    ):
        super(SkorchAttentionNet, self).__init__(
            AttentionNet,
            module__input_dim=input_dim,
            module__model_output_dim=model_output_dim,
            module__hidden_dim=hidden_dim,
            module__output_dim=output_dim,
            *args,
            **kwargs,
        )


class AttentionMetaWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, models, attention_net_params, use_original_X):
        self.models = models
        self.attention_net_params = attention_net_params
        self.use_original_X = use_original_X
        self.attention_net = SkorchAttentionNet(
            **attention_net_params,
            callbacks=[EarlyStopping(patience=20)],
        )

    def fit(self, X, y):
        model_outputs = np.column_stack([model.predict(X) for model in self.models])
        if self.use_original_X:
            combined_input = np.hstack([X, model_outputs])
        else:
            combined_input = model_outputs
        combined_input = combined_input.astype(np.float32)  # Ensure input is float32
        y = y.astype(np.float32).reshape(
            -1, 1
        )  # Ensure target is float32 and reshaped to match predictions
        self.attention_net.fit(combined_input, y)
        return self

    def predict(self, X):
        model_outputs = np.column_stack([model.predict(X) for model in self.models])
        if self.use_original_X:
            combined_input = np.hstack([X, model_outputs])
        else:
            combined_input = model_outputs
        combined_input = combined_input.astype(np.float32)  # Ensure input is float32
        return self.attention_net.predict(combined_input).reshape(-1)


if __name__ == "__main__":
    # Load diabetes dataset
    X, y = load_diabetes(return_X_y=True)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Standardize the features and the target variable
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Define base models
    models = [LinearRegression(), DecisionTreeRegressor()]

    for model in models:
        model.fit(X_train, y_train)

    # Define attention net parameters
    attention_net_params = {
        "input_dim": X_train.shape[1],
        "model_output_dim": len(models),  # Number of base models
        "hidden_dim": 10,
        "output_dim": 1,
        "max_epochs": 500,
        "lr": 0.01,
    }

    # Create and fit the attention meta-wrapper (using original X)
    attention_wrapper_with_X = AttentionMetaWrapper(
        models=models, attention_net_params=attention_net_params, use_original_X=True
    )
    attention_wrapper_with_X.fit(X_train, y_train)

    # Predict using the attention meta-wrapper (using original X)
    attention_predictions_with_X = attention_wrapper_with_X.predict(X_test)
    attention_mse_with_X = mean_squared_error(y_test, attention_predictions_with_X)
    print(f"Attention Meta-Wrapper with Original X MSE: {attention_mse_with_X}")

    # Define stacking model
    stacking_regressor = StackingRegressor(
        estimators=[("lr", models[0]), ("dt", models[1])],
        final_estimator=LinearRegression(),
        cv="prefit",
    )

    # Fit the stacking model
    stacking_regressor.fit(X_train, y_train)

    # Predict using the stacking model
    stacking_predictions = stacking_regressor.predict(X_test)
    stacking_mse = mean_squared_error(y_test, stacking_predictions)
    print(f"Stacking Regressor MSE: {stacking_mse}")
