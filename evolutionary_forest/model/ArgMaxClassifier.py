import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin


class MaxValueClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None):
        # This classifier doesn't need fitting, but we include the method to comply with the sklearn API
        return self

    def predict(self, X):
        # Select the index of the maximum value in each sample
        return np.argmax(X, axis=1)

    def predict_proba(self, X):
        return softmax(X, axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)


# Example usage:
if __name__ == "__main__":
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    classifier = MaxValueClassifier()
    classifier.fit(X)
    predictions = classifier.predict(X)
    print(
        predictions
    )  # Output should be [2, 2, 2] since the max value is in the last column for each row
