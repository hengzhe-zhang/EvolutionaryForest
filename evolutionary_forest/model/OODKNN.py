import numpy as np
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from scipy.special import softmax


def rbf_weights(distances, gamma=1.0):
    # Equivalent to softmax(-gamma * d^2)
    return softmax(-gamma * distances**2, axis=1)


class SkipKNeighborsRegressor(KNeighborsRegressor):
    def predict(self, X):
        """
        If all the nearest neighbors have a distance of zero, it might indicate duplicate points or overfitting issues.
        To address this, this model dynamically skips the nearest neighbor and uses the next closest ones.

        Override the predict method to skip the nearest neighbor dynamically
        if all distances to the first neighbor are zero.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Test samples.

        Returns:
        - y: array-like, shape (n_samples,)
            Predicted target values.
        """
        # Always query n_neighbors + 1 neighbors
        distances, indices = self.kneighbors(X, n_neighbors=self.n_neighbors + 1)

        # Check if the nearest neighbor's distance is zero for all samples
        skip_nearest = np.all(distances[:, 0] == 0)

        if skip_nearest:
            # Exclude the closest neighbor and use the next n_neighbors
            distances = distances[:, 1:]
            indices = indices[:, 1:]
        else:
            # Use only the first n_neighbors, ignoring the last one
            distances = distances[:, :-1]
            indices = indices[:, :-1]

        # Retrieve the targets for the neighbors
        neighbor_targets = self._y[indices]

        # Calculate predictions using weights if applicable
        if self.weights == "distance":
            # Inverse Distance Weighting
            # Closer neighbors get higher weights, while distant neighbors get lower influence.
            with np.errstate(divide="ignore"):  # Handle divide by zero if distance is 0
                weights = 1 / distances
                weights[distances == 0] = 0
            y_pred = np.sum(neighbor_targets * weights, axis=1) / np.sum(
                weights, axis=1
            )
        elif callable(self.weights):
            # Custom Weighting
            weights = self.weights(distances)
            y_pred = np.sum(neighbor_targets * weights, axis=1) / np.sum(
                weights, axis=1
            )
        else:
            y_pred = np.mean(neighbor_targets, axis=1)

        return y_pred


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0
    )

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    # Instantiate and fit the custom regressor
    for n in [3, 5, 10, 15, 20]:
        print(n)
        knn = SkipKNeighborsRegressor(n_neighbors=n, weights="distance")
        knn.fit(X_train, y_train)

        # Predict without skipping the nearest neighbor
        y_pred = knn.predict(X_train)
        print(r2_score(y_train, y_pred))

        y_pred_test = knn.predict(X_test)
        print(r2_score(y_test, y_pred_test))
