import copy
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# Encoder and WeightAssigner as in the previous code
class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout_prob=0.1):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Dropout(dropout_prob),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Dropout(dropout_prob),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return self.fc(x)


class WeightAssigner(nn.Module):
    def __init__(self, latent_dim, num_classifiers, temperature=1.0):
        super(WeightAssigner, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classifiers)
        self.temperature = temperature  # New temperature parameter

    def forward(self, z):
        # Adjust softmax using temperature
        weights = torch.softmax(self.fc(z) / self.temperature, dim=-1)
        return weights
        # return self.fc(z)


# The Meta-Learner class compatible with sklearn
class DESMetaRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        latent_dim=32,
        num_epochs=500,
        lr=0.01,
        lambda_contrastive=0.01,
        lambda_entropy=0.01,
        patience=20,
        verbose=False,
        mode="hybrid",  # Mode selection parameter
        regularization_type="entropy",  # New parameter for regularization type
        use_uniform_weights=False,  # Flag to use uniform weights for testing
        temperature=1,  # New temperature parameter
        dropout_prob=0.2,
        **param,
    ):
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.lambda_contrastive = lambda_contrastive
        self.lambda_entropy = lambda_entropy
        self.patience = patience
        self.verbose = verbose
        self.mode = mode  # Store the selected mode
        self.regularization_type = regularization_type  # Store the regularization type
        self.use_uniform_weights = use_uniform_weights  # Store the uniform weights flag
        self.temperature = temperature  # Store the temperature for WeightAssigner
        self.encoder = None
        self.weight_assigner = None
        self.trained = False
        self.scaler = StandardScaler()
        self.prediction_scaler = StandardScaler()
        self.dropout_prob = dropout_prob

    def fit(self, X, predictions, y, batch_size=32):
        # Standardize X and y
        X = self.scaler.fit_transform(X)
        predictions = self.prediction_scaler.fit_transform(predictions)
        y = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

        # Train-validation split
        (
            X_train,
            X_val,
            predictions_train,
            predictions_val,
            y_train,
            y_val,
        ) = train_test_split(X, predictions, y, test_size=0.2, random_state=0)

        # Convert predictions to tensor format
        predictions_train = torch.tensor(predictions_train, dtype=torch.float32)
        predictions_val = torch.tensor(predictions_val, dtype=torch.float32)

        # Define the input dimensions based on the selected mode
        if self.mode == "original":
            input_dim = X.shape[1]
        elif self.mode == "predicted":
            input_dim = predictions.shape[1]
        elif self.mode == "hybrid":
            input_dim = X.shape[1] + predictions.shape[1]

        # Initialize encoder and weight assigner only if not already trained
        if not self.trained:
            self.encoder = Encoder(input_dim, self.latent_dim, self.dropout_prob)
            self.weight_assigner = WeightAssigner(
                self.latent_dim, predictions.shape[1], self.temperature
            )
        else:
            if self.verbose:
                print("Continuing training with existing model weights.")

        if self.use_uniform_weights:
            if self.verbose:
                print("Using uniform weights, skipping NN training.")
            self.trained = True  # Mark as trained for pipeline testing
            return  # Exit early since we don't need NN training with uniform weights

        # Convert training data to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        # Prepare the combined input for training and validation
        if self.mode == "original":
            train_input = X_train_tensor
            val_input = X_val_tensor
        elif self.mode == "predicted":
            train_input = predictions_train
            val_input = predictions_val
        elif self.mode == "hybrid":
            train_input = torch.cat([X_train_tensor, predictions_train], dim=1)
            val_input = torch.cat([X_val_tensor, predictions_val], dim=1)

        # Training model with early stopping
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.weight_assigner.parameters()),
            lr=self.lr,
        )
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.weight_assigner.train()

            # Batch training
            num_samples = train_input.size(0)
            indices = torch.randperm(num_samples)  # Shuffle data
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                batch_input = train_input[batch_indices]
                batch_predictions = predictions_train[batch_indices]
                batch_y = y_train_tensor[batch_indices]

                sorted_indices = torch.argsort(batch_y.view(-1))
                batch_input = batch_input[sorted_indices]
                batch_predictions = batch_predictions[sorted_indices]
                batch_y = batch_y[sorted_indices]

                optimizer.zero_grad()

                # Encode combined input to get latent representation
                z = self.encoder(batch_input)  # Latent representation
                weights = self.weight_assigner(
                    z
                )  # Weights assigned to each base learner prediction

                # Compute weighted predictions using base learner predictions
                weighted_preds = torch.sum(
                    weights * batch_predictions, dim=1, keepdim=True
                )
                des_loss = nn.functional.mse_loss(weighted_preds, batch_y)

                if self.verbose:
                    baseline_loss = nn.functional.mse_loss(
                        batch_predictions.mean(dim=1, keepdim=True), batch_y
                    )
                    print(
                        f"DES Loss: {des_loss.item():.4f}, Baseline Loss: {baseline_loss.item():.4f}"
                    )

                # Contrastive loss (InfoNCE)
                if z.size(0) > 1:
                    contrastive_loss = self._info_nce_loss(z[:-1], z[1:])
                else:
                    contrastive_loss = torch.tensor(0.0)

                # Regularization based on the selected type
                regularization_loss = self._compute_regularization_loss(weights)

                total_loss = (
                    des_loss
                    + self.lambda_contrastive * contrastive_loss
                    + self.lambda_entropy * regularization_loss
                )
                total_loss.backward()
                optimizer.step()

            # Validation loss
            self.encoder.eval()
            self.weight_assigner.eval()
            with torch.no_grad():
                z_val = self.encoder(val_input)
                weights_val = self.weight_assigner(z_val)
                weighted_preds_val = torch.sum(
                    weights_val * predictions_val, dim=1, keepdim=True
                )
                val_loss = nn.functional.mse_loss(weighted_preds_val, y_val_tensor)

                if self.verbose:
                    print(
                        f"Epoch: {epoch + 1}, Total Loss: {total_loss.item():.4f}, Validation Loss: {val_loss.item():.4f}"
                    )

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = (
                    copy.deepcopy(self.encoder.state_dict()),
                    copy.deepcopy(self.weight_assigner.state_dict()),
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print("Early stopping at epoch:", epoch + 1)
                    break

        # Load best model weights
        if best_state:
            self.encoder.load_state_dict(best_state[0])
            self.weight_assigner.load_state_dict(best_state[1])

        # Mark the model as trained for future continual learning
        self.trained = True

    def _compute_regularization_loss(self, weights):
        if self.regularization_type == "cosine":
            regularization_loss = self._cosine_regularization_loss(weights)
        else:
            regularization_loss = -torch.sum(
                weights * torch.log(weights + 1e-10)
            ) / weights.size(0)
        return regularization_loss

    def _cosine_regularization_loss(self, weights):
        normalized_weights = F.normalize(weights, p=2, dim=1)
        cosine_sim_matrix = torch.matmul(normalized_weights.T, normalized_weights)
        num_weights = weights.size(1)
        mask = torch.eye(num_weights, device=weights.device).bool()
        cosine_sim_matrix = cosine_sim_matrix.masked_fill(mask, 0)
        diversity_penalty = cosine_sim_matrix.sum() / (num_weights * (num_weights - 1))
        regularization_loss = diversity_penalty
        return regularization_loss

    def predict(self, X_meta, predictions, batch_size=32):
        # Standardize inputs using the scaler fitted in training
        X_meta = self.scaler.transform(X_meta)
        predictions = self.prediction_scaler.transform(predictions)

        if self.use_uniform_weights:
            # Use uniform weights if the flag is set
            uniform_weights = np.ones(predictions.shape[1]) / predictions.shape[1]
            weighted_preds = (
                uniform_weights * self.prediction_scaler.inverse_transform(predictions)
            ).sum(axis=1)
            return weighted_preds

        self.encoder.eval()
        self.weight_assigner.eval()

        X_tensor = torch.tensor(X_meta, dtype=torch.float32)
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)

        # Prepare the combined input for prediction based on mode
        if self.mode == "original":
            combined_input = X_tensor
        elif self.mode == "predicted":
            combined_input = predictions_tensor
        elif self.mode == "hybrid":
            combined_input = torch.cat([X_tensor, predictions_tensor], dim=1)

        # Batch prediction
        num_samples = combined_input.size(0)
        predictions_list = []

        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_input = combined_input[start:end]

                # Obtain latent representations and calculate weights
                z = self.encoder(batch_input)
                weights = self.weight_assigner(z)

                # Combine provided predictions using weights
                batch_predictions = (
                    weights.numpy()
                    * self.prediction_scaler.inverse_transform(predictions[start:end])
                ).sum(axis=1)
                predictions_list.append(batch_predictions)

        # Concatenate batch predictions
        return np.concatenate(predictions_list)

    def _info_nce_loss(self, z_i, z_j, temperature=0.5):
        z_i = nn.functional.normalize(z_i, dim=-1)
        z_j = nn.functional.normalize(z_j, dim=-1)
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
        neg_sim = torch.exp(torch.matmul(z_i, z_j.T) / temperature)
        neg_sim = torch.sum(neg_sim, dim=-1) - pos_sim
        loss = -torch.log(pos_sim / neg_sim).mean()
        return loss

    def enable_training_mode(self):
        self.encoder.train()
        self.weight_assigner.train()

    def enable_evaluation_mode(self):
        self.encoder.eval()
        self.weight_assigner.eval()

    def count_base_learner_usage(self, X_meta, predictions, batch_size=32, mode="sum"):
        # Standardize inputs using the scaler fitted during training
        X_meta = self.scaler.transform(X_meta)
        predictions = self.prediction_scaler.transform(predictions)

        self.encoder.eval()
        self.weight_assigner.eval()

        # Prepare tensors
        X_tensor = torch.tensor(X_meta, dtype=torch.float32)
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)

        # Prepare the combined input based on the mode
        if self.mode == "original":
            combined_input = X_tensor
        elif self.mode == "predicted":
            combined_input = predictions_tensor
        elif self.mode == "hybrid":
            combined_input = torch.cat([X_tensor, predictions_tensor], dim=1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        num_samples = combined_input.size(0)
        total_weights = None

        # Accumulate weights across all batches
        with torch.no_grad():
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_input = combined_input[start:end]

                # Obtain latent representations and calculate weights
                z = self.encoder(batch_input)
                weights = self.weight_assigner(z)  # Shape: (batch_size, n_models)

                # Sum weights for the current batch
                if total_weights is None:
                    total_weights = weights.sum(dim=0)
                else:
                    total_weights += weights.sum(dim=0)

        # Convert total weights to numpy
        total_weights = total_weights.numpy()

        # Aggregate based on mode
        if mode == "sum":
            return total_weights
        elif mode == "average":
            return total_weights / num_samples
        else:
            raise ValueError(f"Invalid aggregation mode: {mode}")

    def plot_sample_weights(self, X_meta_samples, predictions_samples, mode="hybrid"):
        """Plots the weights assigned to each model for each sample in X_meta_samples."""
        self.encoder.eval()
        self.weight_assigner.eval()

        # Prepare the input tensor based on the mode
        if mode == "original":
            X_tensor = torch.tensor(X_meta_samples, dtype=torch.float32)
        elif mode == "predicted":
            X_tensor = torch.tensor(predictions_samples, dtype=torch.float32)
        elif mode == "hybrid":
            X_tensor = torch.cat(
                [
                    torch.tensor(X_meta_samples, dtype=torch.float32),
                    torch.tensor(predictions_samples, dtype=torch.float32),
                ],
                dim=1,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        with torch.no_grad():
            z = self.encoder(X_tensor)
            weights = self.weight_assigner(z)

        # Convert weights to numpy for plotting
        weights_np = weights.numpy()

        # Plot heatmap
        plt.figure(
            figsize=(10, min(1 + weights_np.shape[0] * 0.4, 12))
        )  # Adaptive height for number of samples
        sns.heatmap(
            weights_np,
            annot=False,
            cmap="viridis",
            cbar=True,
            xticklabels=True,
            yticklabels=True,
        )
        plt.xlabel("Model Index")
        plt.ylabel("Sample Index")
        plt.title("Model Weights Heatmap for Multiple Samples")
        plt.show()


# Updated run_experiment function for parallel processing
def run_experiment(args):
    (
        params,
        X_train,
        predictions_train,
        y_train,
        X_val,
        predictions_val,
        y_val,
    ) = args

    # Create DESMetaRegressor instance using the parameter dictionary
    des_regressor = DESMetaRegressor(**params)

    # Fit the model with X, predictions, and y
    des_regressor.fit(X_train, predictions_train, y_train)

    # Predict on the training and validation sets
    y_pred_train = des_regressor.predict(
        X_train,
        predictions_train,
    )
    y_pred_val = des_regressor.predict(
        X_val,
        predictions_val,
    )

    # Calculate R2 scores for training and validation sets
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)

    # Return parameters and R2 scores wrapped in a dictionary
    return params, {"r2_train": r2_train, "r2_val": r2_val}


# Main function for hyperparameter tuning using sklearn's ParameterGrid
def run_hyperparameters(
    X_train, predictions_train, y_train, X_val, predictions_val, y_val, param_grid
):
    # Generate tasks for each combination in ParameterGrid
    tasks = [
        (
            params,
            X_train,
            predictions_train,
            y_train,
            X_val,
            predictions_val,
            y_val,
        )
        for params in ParameterGrid(param_grid)
    ]

    # Use ProcessPoolExecutor to parallelize hyperparameter tuning
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_experiment, tasks))

    return results


# Function to handle hyperparameter tuning setup and execution
def run_parameter_tuning(base_learners, X_train, y_train, X_val, y_val):
    # Generate predictions from base learners for training and validation sets
    predictions_train = np.column_stack(
        [learner.predict(X_train) for learner in base_learners]
    )
    predictions_val = np.column_stack(
        [learner.predict(X_val) for learner in base_learners]
    )

    # Define the parameter grid
    param_grid = {
        "lambda_contrastive": [0.01, 0.1],
        "lr": [0.01],
        "lambda_entropy": [0.01, 0.1],
        "mode": ["hybrid"],
        "temperature": [1],
        "regularization_type": ["entropy", "cosine"],
        "use_uniform_weights": [False],
    }

    # Run hyperparameter tuning with generated predictions
    results = run_hyperparameters(
        X_train, predictions_train, y_train, X_val, predictions_val, y_val, param_grid
    )

    # Print results
    print("Hyperparameter tuning results:")
    for params, scores in results:
        print(f"Params: {params}, Scores: {scores}")


def evaluate_base_learners_and_average(base_learners, X_val, y_val):
    """Evaluates each base learner individually and calculates the simple averaging performance on validation set."""
    # Predict with each base learner and calculate individual performance
    base_learner_preds = []
    print("Base Learner Performance on Validation Set:")
    for i, learner in enumerate(base_learners):
        y_pred = learner.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        print(f"Base Learner {i + 1} ({learner.__class__.__name__}) R2: {r2:.4f}")
        base_learner_preds.append(y_pred.reshape(-1, 1))

    # Calculate simple averaging performance
    base_learner_preds = np.hstack(base_learner_preds)
    y_pred_avg = np.mean(base_learner_preds, axis=1)
    r2_avg = r2_score(y_val, y_pred_avg)
    print(f"\nSimple Averaging of Base Learners R2 on Validation Set: {r2_avg:.4f}")

    return r2_avg


# Function to display weights assigned to each model for multiple samples as a heatmap
def show_model_weights_heatmap(
    des_model, X_meta_samples, predictions_samples, mode="hybrid", num_samples=20
):
    """Displays a heatmap of weights assigned by DESMetaRegressor to each model for multiple samples."""
    des_model.encoder.eval()
    des_model.weight_assigner.eval()

    X_meta_samples = des_model.scaler.transform(X_meta_samples)
    predictions_samples = des_model.prediction_scaler.transform(predictions_samples)

    # Prepare the input tensor based on the mode
    if mode == "original":
        X_tensor = torch.tensor(X_meta_samples, dtype=torch.float32)
    elif mode == "predicted":
        X_tensor = torch.tensor(predictions_samples, dtype=torch.float32)
    elif mode == "hybrid":
        X_tensor = torch.cat(
            [
                torch.tensor(X_meta_samples, dtype=torch.float32),
                torch.tensor(predictions_samples, dtype=torch.float32),
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    with torch.no_grad():
        z = des_model.encoder(X_tensor)
        weights = des_model.weight_assigner(z)

    # Convert weights to numpy array for easier visualization
    weights_np = weights.numpy()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(weights_np, aspect="auto", cmap="viridis")
    plt.colorbar(label="Weight")
    plt.xlabel("Base Learner Model Index")
    plt.ylabel("Sample Index")
    plt.title(f"Model Weights Heatmap for {num_samples} Samples")
    plt.show()


# Function to use validation data and display model weights heatmap for selected samples
def display_sample_weights_on_validation(
    des_model, base_learners, X_val, num_samples=20, mode="hybrid"
):
    """Uses the validation data to display weights heatmap for a few samples."""
    # Generate meta-features for the validation data using the base learners
    predictions_val = np.column_stack(
        [learner.predict(X_val) for learner in base_learners]
    )

    # Select a few samples to display weights
    sample_indices = np.random.choice(X_val.shape[0], num_samples, replace=False)
    X_meta_samples = X_val[sample_indices]
    predictions_samples = predictions_val[sample_indices]

    # Display heatmap of weights for selected samples
    print(
        f"\nShowing model weights heatmap for {num_samples} random validation samples:"
    )
    show_model_weights_heatmap(
        des_model, X_meta_samples, predictions_samples, mode=mode
    )


# Main function to initialize and visualize weights
def weight_visualization(
    base_learners, X_train, y_train, X_meta_train, X_val, y_val, mode="hybrid"
):
    # Initialize and train DESMetaRegressor with the specified mode
    des_model = DESMetaRegressor(mode=mode)
    des_model.fit(X_train, X_meta_train, y_train)

    # Display weights for a few samples from the validation set
    display_sample_weights_on_validation(
        des_model, base_learners, X_val, num_samples=10, mode=mode
    )


if __name__ == "__main__":
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Define base learners and obtain meta features
    base_learners = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=10, random_state=0),
        GradientBoostingRegressor(n_estimators=10, random_state=0),
        KNeighborsRegressor(n_neighbors=5),
        KNeighborsRegressor(n_neighbors=3),
    ]

    kf = KFold(n_splits=5)
    oof_preds = []
    for learner in base_learners:
        preds = cross_val_predict(learner, X_train, y_train, cv=kf)
        learner.fit(X_train, y_train)
        oof_preds.append(preds.reshape(-1, 1))

    X_meta_train = np.hstack(oof_preds)
    X_meta_val = np.column_stack([learner.predict(X_val) for learner in base_learners])

    # weight_visualization(base_learners, X_train, y_train, X_val, y_val, mode="original")

    # r2_avg = evaluate_base_learners_and_average(base_learners, X_val, y_val)

    # Run parameter tuning
    run_parameter_tuning(base_learners, X_train, y_train, X_val, y_val)
