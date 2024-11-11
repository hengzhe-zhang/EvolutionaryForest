class GPWrapper:
    def __init__(
        self, individual, feature_generation_function, y_scaler, x_scaler=None
    ):
        self.individual = individual
        self.feature_generation_function = feature_generation_function
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def fit(self, X, y):
        pass

    def predict(self, X):
        if self.x_scaler is not None:
            X = self.x_scaler.transform(X)
        features = self.feature_generation_function(X, self.individual)
        predicted = self.individual.pipe.predict(features)
        if self.y_scaler is not None:
            predicted = self.y_scaler.inverse_transform(predicted.reshape(-1, 1))
        return predicted.flatten()
