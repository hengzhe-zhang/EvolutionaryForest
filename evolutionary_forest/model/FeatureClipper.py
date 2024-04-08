import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from evolutionary_forest.model.SafetyScaler import SafetyScaler


class FeatureClipper(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Initialize the StandardScaler
        self.scaler = SafetyScaler()
        # Initialize min and max values for clipping
        self.feature_mins_ = None
        self.feature_maxs_ = None

    def fit(self, X, y=None):
        # Fit the StandardScaler to the data
        self.scaler.fit(X)
        # Transform the data using the fitted StandardScaler
        X_scaled = self.scaler.transform(X)
        # Learn the minimum and maximum values for each feature from the scaled data
        self.feature_mins_ = np.min(X_scaled, axis=0)
        self.feature_maxs_ = np.max(X_scaled, axis=0)
        return self  # Return self to allow chaining

    def transform(self, X):
        # First, standardize the features using the fitted StandardScaler
        X_standardized = self.scaler.transform(X)
        # Then, clip the standardized features to the min and max values learned during fitting
        X_clipped = np.clip(X_standardized, self.feature_mins_, self.feature_maxs_)
        return X_clipped
