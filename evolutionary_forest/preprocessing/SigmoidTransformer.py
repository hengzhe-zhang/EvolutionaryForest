from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


class SigmoidTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler((1e-3, 1 - 1e-3))

    def sigmoid(self, x):
        return expit(x)

    def inverse_sigmoid(self, x):
        return logit(x)

    def fit(self, X, y=None):
        # Fit method doesn't do anything in this case
        self.scaler.fit(X.reshape(-1, 1))
        return self

    def transform(self, X):
        X = self.scaler.transform(X.reshape(-1, 1)).flatten()
        return self.inverse_sigmoid(X)

    def inverse_transform(self, X):
        X = self.sigmoid(X)
        return self.scaler.inverse_transform(X.reshape(-1, 1)).flatten()


if __name__ == "__main__":
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=100, random_state=0
    )

    # Create a LinearRegression model
    lr = SVR()

    # Fit the model on the training data
    lr.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = lr.predict(X_test)

    # Calculate and print the Mean Squared Error (MSE) without sigmoid transformation
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE) without sigmoid transformation: {mse:.2f}")

    # Create a SigmoidTransformer instance to transform the target variable y_train
    sigmoid_transformer = SigmoidTransformer()
    y_train_transformed = sigmoid_transformer.fit_transform(y_train)

    # Fit a LinearRegression model on the transformed target variable
    linear_regression = SVR()
    linear_regression.fit(X_train, y_train_transformed)

    # Make predictions on the test data with the transformed target variable
    y_pred_transformed = linear_regression.predict(X_test)

    # Inverse transform the predictions to get back to the original range
    y_pred_inverse = sigmoid_transformer.inverse_transform(y_pred_transformed)

    # Calculate and print the Mean Squared Error (MSE) with sigmoid transformation
    mse_transformed = mean_squared_error(y_test, y_pred_inverse)
    print(
        f"Mean Squared Error (MSE) with sigmoid transformation: {mse_transformed:.2f}"
    )
