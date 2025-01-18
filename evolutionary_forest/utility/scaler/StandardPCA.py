from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class StandardScalerPCA(TransformerMixin):
    def __init__(self, n_components=None):
        """
        A class that wraps StandardScaler and PCA from scikit-learn.

        Parameters:
        - n_components: int, float, None or str
            The number of components to keep. If n_components is not set or None, all components are kept.
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components

    def fit(self, X, y=None):
        """
        Fit the model with X by computing the mean and standard deviation for scaling
        and fitting the PCA on the scaled data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            The data to fit.
        - y: Ignored
            Not used, present here for API consistency by convention.

        Returns:
        self : object
            Returns the instance itself.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        return self

    def transform(self, X, y=None):
        """
        Scale and apply PCA transformation to the input data.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns:
        X_new : array-like, shape (n_samples, n_components)
            The transformed data.
        """
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return X_pca

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            The data to fit and transform.

        Returns:
        X_new : array-like, shape (n_samples, n_components)
            The transformed data.
        """
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        return X_pca


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=0
    )

    # Print the shape of the original dataset
    print(f"Original dataset shape: {X.shape}")
    # Instantiate the StandardScalerPCA with 10 components
    scaler_pca = StandardScalerPCA(n_components=0.99)

    # Fit the scaler to the dataset
    scaler_pca.fit(X)

    # Transform the dataset
    X_transformed = scaler_pca.transform(X)

    # Print the shape of the transformed dataset
    print(f"Transformed dataset shape: {X_transformed.shape}")
