import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeRegressor

from utils.windows_notification_util import windows_notification


class AverageEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressors_list=None):
        self.regressors_list = regressors_list

    def fit(self, X, y):
        for regressor in self.regressors_list:
            regressor.fit(X, y)
        return self

    def predict(self, X):
        predictions = [regressor.predict(X) for regressor in self.regressors_list]
        return np.mean(predictions, axis=0)


def train_base_regressors(X_train, y_train):
    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor().fit(X_train, y_train)
    rf = RandomForestRegressor().fit(X_train, y_train)
    return [lr, dt, rf]


def split_train_validation(X_train, y_train):
    """
    Splits the training data into training and validation sets.
    """
    return train_test_split(X_train, y_train, test_size=0.5, random_state=0)


class DynamicEnsembleSelectionRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        regressors_list=None,
        competence_region="knn",
        k=5,
        aggregation_method="mean",
        competence_metric=mean_squared_error,
        random_state=None,
        n_jobs=-1,
    ):
        self.regressors_list = regressors_list
        self.competence_region = competence_region
        self.k = k
        self.aggregation_method = aggregation_method
        self.competence_metric = competence_metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X_val, y_val):
        """
        Fit the model using the validation data only.
        """
        self.X_val = X_val
        self.y_val = y_val
        if self.competence_region == "cluster":
            self.cluster_model = KMeans(
                n_clusters=self.k, random_state=self.random_state
            )
            self.cluster_model.fit(X_val)  # Clustering is based on the validation set
        return self

    def _get_competence_region(self, x):
        """
        Identify the competence region for the test sample.
        """
        if self.competence_region == "knn":
            nn = NearestNeighbors(n_neighbors=self.k, n_jobs=self.n_jobs)
            nn.fit(self.X_val)  # Competence region is based on the validation set
            distances, indices = nn.kneighbors([x])
            return indices[0], distances[0]
        elif self.competence_region == "cluster":
            cluster_label = self.cluster_model.predict([x])[0]
            cluster_indices = np.where(self.cluster_model.labels_ == cluster_label)[0]
            return cluster_indices, None

    def _calculate_competence_levels(self, indices):
        """
        Calculate competence levels using the validation set.
        """
        region_X = self.X_val[indices]
        region_y = self.y_val[indices]
        competence_levels = []
        for regressor in self.regressors_list:
            predictions = regressor.predict(region_X)
            error = self.competence_metric(region_y, predictions)
            competence_levels.append(1 / (error + 1e-8))  # Inverse of error
        return np.array(competence_levels)

    def _select_subset(self, competence_levels):
        """
        Select regressors with competence levels above the mean.
        """
        mean_level = np.mean(competence_levels)
        selected_indices = competence_levels >= mean_level
        return selected_indices

    def predict(self, X):
        """
        Predict using the dynamic ensemble.
        """
        predictions = []
        for x in X:
            indices, _ = self._get_competence_region(x)
            competence_levels = self._calculate_competence_levels(indices)
            subset_mask = self._select_subset(competence_levels)
            competence_levels = competence_levels[subset_mask]
            normalized_competence = competence_levels / competence_levels.sum()

            selected_regressors = [
                regressor
                for i, regressor in enumerate(self.regressors_list)
                if subset_mask[i]
            ]

            final_prediction = 0
            for i, regressor in enumerate(selected_regressors):
                pred = regressor.predict([x])[0]
                weight = (
                    normalized_competence[i]
                    if self.aggregation_method == "weighted"
                    else 1 / len(selected_regressors)
                )
                final_prediction += weight * pred
            predictions.append(final_prediction)
        return np.array(predictions)


def manual_grid_search(X_val, y_val, regressors):
    """
    Perform manual grid search using validation set for competence scores.
    """
    param_grid = {
        "k": [3, 5, 7],
        "competence_region": ["knn", "cluster"],
        "aggregation_method": ["mean", "weighted"],
    }
    best_params = None
    best_score = -np.inf

    for k in param_grid["k"]:
        for competence_region in param_grid["competence_region"]:
            for aggregation_method in param_grid["aggregation_method"]:
                des = DynamicEnsembleSelectionRegressor(
                    regressors_list=regressors,
                    k=k,
                    competence_region=competence_region,
                    aggregation_method=aggregation_method,
                )
                des.fit(X_val, y_val)
                y_pred = des.predict(X_val)
                score = r2_score(y_val, y_pred)
                print(
                    f"Params: k={k}, competence_region={competence_region}, "
                    f"aggregation_method={aggregation_method}, R2={score:.4f}"
                )
                if score > best_score:
                    best_score = score
                    best_params = {
                        "k": k,
                        "competence_region": competence_region,
                        "aggregation_method": aggregation_method,
                    }

    print("Best Params:", best_params)
    print("Best R2 Score:", best_score)
    return best_params


def generate_dataset():
    # X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=0)
    X, y = load_diabetes(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=0)


@windows_notification
def run_workflow():
    # Generate Data
    X_train_full, X_test, y_train_full, y_test = generate_dataset()
    # Split training into training and validation sets
    X_train, X_val, y_train, y_val = split_train_validation(X_train_full, y_train_full)
    regressors = train_base_regressors(X_train, y_train)
    # Perform Grid Search
    best_params = manual_grid_search(X_val, y_val, regressors)
    # Train DES on validation set and evaluate on the test set
    des = DynamicEnsembleSelectionRegressor(
        regressors_list=regressors,
        k=best_params["k"],
        competence_region=best_params["competence_region"],
        aggregation_method=best_params["aggregation_method"],
    )
    des.fit(X_val, y_val)
    y_pred_des = des.predict(X_test)
    # Evaluate Average Ensemble
    avg_ensemble = AverageEnsembleRegressor(regressors_list=regressors)
    avg_ensemble.fit(X_train_full, y_train_full)
    y_pred_avg = avg_ensemble.predict(X_test)
    # Results
    train_r2_des = r2_score(y_train_full, des.predict(X_train_full))
    train_r2_avg = r2_score(y_train_full, avg_ensemble.predict(X_train_full))
    print(f"Training R2 Score (DES): {train_r2_des:.4f}")
    print(f"Training R2 Score (Average Ensemble): {train_r2_avg:.4f}")
    r2_des = r2_score(y_test, y_pred_des)
    r2_avg = r2_score(y_test, y_pred_avg)
    print(f"R2 Score (DES): {r2_des:.4f}")
    print(f"R2 Score (Average Ensemble): {r2_avg:.4f}")


if __name__ == "__main__":
    run_workflow()
