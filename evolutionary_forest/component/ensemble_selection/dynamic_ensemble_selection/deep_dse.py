import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# Encoder and WeightAssigner as in the previous code
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        return self.fc(x)


class WeightAssigner(nn.Module):
    def __init__(self, latent_dim, num_classifiers):
        super(WeightAssigner, self).__init__()
        self.fc = nn.Linear(latent_dim, num_classifiers)

    def forward(self, z):
        weights = torch.softmax(self.fc(z), dim=-1)  # Ensures weights sum to 1
        return weights


# The Meta-Learner class compatible with sklearn
class DESMetaRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        latent_dim=5,
        num_epochs=50,
        lr=0.001,
        lambda_contrastive=0.1,
        lambda_entropy=0.01,  # New regularization parameter for entropy
        patience=5,
        verbose=False,
    ):
        self.latent_dim = latent_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.lambda_contrastive = lambda_contrastive
        self.lambda_entropy = lambda_entropy  # Entropy regularization weight
        self.patience = patience
        self.verbose = verbose  # Toggle training log printing
        self.encoder = None
        self.weight_assigner = None
        self.scaler = StandardScaler()  # Scaler for X
        self.y_scaler = StandardScaler()  # Scaler for y

    def fit(self, X, y):
        # Standardize the input data
        X = self.scaler.fit_transform(X)
        y = self.y_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel()  # Standardize y and flatten

        # Initialize encoder and weight assigner after knowing the input dimension
        input_dim = X.shape[1]
        self.encoder = Encoder(input_dim, self.latent_dim)
        self.weight_assigner = WeightAssigner(
            self.latent_dim, X.shape[1]
        )  # Number of base learners

        # Convert data to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Training model with early stopping
        optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.weight_assigner.parameters()),
            lr=self.lr,
        )
        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.weight_assigner.train()
            optimizer.zero_grad()

            z = self.encoder(X_tensor)  # Latent representation
            weights = self.weight_assigner(z)  # Softmax weights

            # Convert predictions to tensor format
            ensemble_preds_tensor = torch.tensor(X, dtype=torch.float32)

            # Compute DES regression loss
            weighted_preds = torch.sum(
                weights * ensemble_preds_tensor, dim=1, keepdim=True
            )
            des_loss = nn.functional.mse_loss(weighted_preds, y_tensor)

            # Contrastive loss (InfoNCE)
            if z.size(0) > 1:  # Ensure batch size > 1 for contrastive pairs
                contrastive_loss = self._info_nce_loss(z[:-1], z[1:])
            else:
                contrastive_loss = torch.tensor(0.0)

            # Entropy regularization to reduce overconfidence
            entropy_loss = -torch.sum(
                weights * torch.log(weights + 1e-10)
            ) / weights.size(0)

            # Total loss with entropy regularization
            total_loss = (
                des_loss
                + self.lambda_contrastive * contrastive_loss
                + self.lambda_entropy * entropy_loss
            )
            total_loss.backward()
            optimizer.step()

            if self.verbose:
                print(
                    f"Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss.item():.4f}, "
                    f"DES Loss: {des_loss.item():.4f}, Contrastive Loss: {contrastive_loss.item():.4f}, "
                    f"Entropy Loss: {entropy_loss.item():.4f}"
                )

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
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
        self.encoder.load_state_dict(best_state[0])
        self.weight_assigner.load_state_dict(best_state[1])

    def predict(self, X_meta):
        # Standardize the input data using the scaler fitted in training
        X_meta = self.scaler.transform(X_meta)

        self.encoder.eval()
        self.weight_assigner.eval()
        X_tensor = torch.tensor(X_meta, dtype=torch.float32)

        with torch.no_grad():
            z = self.encoder(X_tensor)
            weights = self.weight_assigner(z)
            ensemble_preds_tensor = torch.tensor(
                self.scaler.inverse_transform(X_meta), dtype=torch.float32
            )
            weighted_preds = torch.sum(weights * ensemble_preds_tensor, dim=1)

        return weighted_preds.numpy()

    def _info_nce_loss(self, z_i, z_j, temperature=0.5):
        z_i = nn.functional.normalize(z_i, dim=-1)
        z_j = nn.functional.normalize(z_j, dim=-1)
        pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
        neg_sim = torch.exp(torch.matmul(z_i, z_j.T) / temperature)
        neg_sim = (
            torch.sum(neg_sim, dim=-1) - pos_sim
        )  # Exclude positive from denominator
        loss = -torch.log(pos_sim / neg_sim).mean()
        return loss


# Run experiment function at top level for parallel processing
def run_experiment(args):
    lambda_contrastive, lambda_entropy, X_meta_train, y_train, X_meta_val, y_val = args
    des_regressor = DESMetaRegressor(
        latent_dim=5,
        lambda_contrastive=lambda_contrastive,
        lambda_entropy=lambda_entropy,
        num_epochs=500,
        patience=10,
        verbose=False,
    )
    des_regressor.fit(X_meta_train, y_train)
    y_pred_des = des_regressor.predict(X_meta_val)
    r2 = r2_score(y_val, y_pred_des)
    return lambda_contrastive, lambda_entropy, r2


def run_hyperparameters(
    X_meta_train,
    y_train,
    X_meta_val,
    y_val,
    lambda_contrastive_values,
    lambda_entropy_values,
):
    tasks = [
        (lambda_contrastive, lambda_entropy, X_meta_train, y_train, X_meta_val, y_val)
        for lambda_contrastive in lambda_contrastive_values
        for lambda_entropy in lambda_entropy_values
    ]

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment, tasks))

    return results


def run_parameter_tuning():
    # Define ranges for lambda_contrastive and lambda_entropy
    lambda_contrastive_values = [1.0, 10]
    lambda_entropy_values = [0.001, 0]

    # Run hyperparameter tuning
    results = run_hyperparameters(
        X_meta_train,
        y_train,
        X_meta_val,
        y_val,
        lambda_contrastive_values,
        lambda_entropy_values,
    )

    # Print results
    print("Hyperparameter tuning results:")
    for lambda_contrastive, lambda_entropy, r2 in results:
        print(
            f"lambda_contrastive: {lambda_contrastive}, lambda_entropy: {lambda_entropy}, R2: {r2:.4f}"
        )


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


# Add function to show weights assigned to each model on a few samples
def show_model_weights(des_model, X_meta_samples):
    """Displays weights assigned by DESMetaRegressor to each model for given samples."""
    des_model.encoder.eval()
    des_model.weight_assigner.eval()
    X_tensor = torch.tensor(X_meta_samples, dtype=torch.float32)

    with torch.no_grad():
        z = des_model.encoder(X_tensor)
        weights = des_model.weight_assigner(z)

    weights_np = weights.numpy()  # Convert to numpy for easier inspection
    for i, weight_vector in enumerate(weights_np):
        print(f"Sample {i + 1} - Weights: {weight_vector}")


# Add function to use validation data and display model weights on selected samples
def display_sample_weights_on_validation(
    des_model, base_learners, X_val, num_samples=5
):
    """Uses the validation data to display weights for a few samples."""
    # Generate meta-features for the validation data using the base learners
    X_meta_val = np.column_stack([learner.predict(X_val) for learner in base_learners])

    # Select a few samples to display weights
    sample_indices = np.random.choice(X_meta_val.shape[0], num_samples, replace=False)
    X_meta_samples = X_meta_val[sample_indices]

    print(f"\nShowing model weights for {num_samples} random validation samples:")
    show_model_weights(des_model, X_meta_samples)


if __name__ == "__main__":
    data = load_diabetes()
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define base learners and obtain meta features
    base_learners = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=10, random_state=42),
        KNeighborsRegressor(n_neighbors=5),
    ]

    kf = KFold(n_splits=5)
    oof_preds = []
    for learner in base_learners:
        preds = cross_val_predict(learner, X_train, y_train, cv=kf)
        learner.fit(X_train, y_train)
        oof_preds.append(preds.reshape(-1, 1))

    X_meta_train = np.hstack(oof_preds)
    X_meta_val = np.column_stack([learner.predict(X_val) for learner in base_learners])

    # # Initialize and train DESMetaRegressor
    # des_model = DESMetaRegressor(
    #     latent_dim=10,
    #     num_epochs=100,
    #     patience=10,
    #     lambda_contrastive=0.1,
    #     lambda_entropy=0.1,
    #     verbose=True,
    # )
    # des_model.fit(X_meta_train, y_train)
    #
    # # Display weights for a few samples from the validation set
    # display_sample_weights_on_validation(des_model, base_learners, X_val, num_samples=5)

    r2_avg = evaluate_base_learners_and_average(base_learners, X_val, y_val)

    # Run parameter tuning
    # run_parameter_tuning()
