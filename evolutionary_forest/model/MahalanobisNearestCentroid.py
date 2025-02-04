import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import cdist


class MahalanobisNearestCentroid(BaseEstimator, ClassifierMixin):
    def __init__(self, standardize=True):
        """
        Nearest Centroid Classifier with Mahalanobis Distance (Vectorized).
        """
        self.standardize = standardize
        self.centroids_ = None
        self.inv_cov_matrix_ = None
        self.classes_ = None
        self.scaler_ = None

    def fit(self, X, y):
        """
        Fit the classifier by computing class centroids and the inverse covariance matrix.
        """
        self.classes_ = np.unique(y)

        # Standardize features if enabled
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # Compute centroids
        self.centroids_ = np.array(
            [X[y == label].mean(axis=0) for label in self.classes_]
        )

        # Compute inverse covariance matrix
        cov_matrix = np.cov(X.T)
        self.inv_cov_matrix_ = np.linalg.inv(cov_matrix)

        return self

    def predict(self, X):
        """
        Predict class labels using vectorized Mahalanobis distance computation.
        """
        if self.standardize and self.scaler_:
            X = self.scaler_.transform(X)

        # Use cdist for efficient Mahalanobis distance computation
        distances = cdist(
            X, self.centroids_, metric="mahalanobis", VI=self.inv_cov_matrix_
        )

        # Assign each test instance to the class with the smallest distance
        return self.classes_[np.argmin(distances, axis=1)]


if __name__ == "__main__":
    from sklearn.datasets import make_circles
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Generate synthetic dataset
    # X, y = load_iris(return_X_y=True)
    X, y = make_circles()

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Create and train classifier
    clf = MahalanobisNearestCentroid()
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Create and train classifier
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
