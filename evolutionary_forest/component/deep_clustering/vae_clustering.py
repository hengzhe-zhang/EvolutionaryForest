import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder: outputs mean and log variance
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(16, latent_dim)  # Log variance of latent space

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def encode(self, x):
        """Encodes the input to a latent space (mu and logvar)."""
        h = self.encoder_fc(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, sigma^2) using N(0, 1)."""
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)  # Random normal noise
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        """Decodes the latent space variable z back to the original input space."""
        return self.decoder(z)

    def forward(self, x):
        # Encode input to latent space
        mu, logvar = self.encode(x)
        # Reparameterize to get the latent variable z
        z = self.reparameterize(mu, logvar)
        # Decode back to original input space
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Computes the VAE loss, which includes reconstruction and KL divergence using MSE."""
        # Reconstruction loss (mean squared error for continuous data)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

        # KL Divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss = reconstruction loss + KL divergence loss
        return recon_loss + kld_loss


class DeepClustering(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters=10,
        latent_dim=10,
        lambda_=0.1,
        batch_size=256,
        epochs=50,
        pretrain_epochs=10,
        lr=1e-3,
        entropy_lambda=0,  # Entropy regularization weight
        balance_lambda=0.1,  # Cluster balance regularization weight
        patience=10,  # Early stopping patience
        device=None,
    ):
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.lambda_ = lambda_
        self.entropy_lambda = entropy_lambda
        self.balance_lambda = balance_lambda
        self.batch_size = batch_size
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.lr = lr
        self.patience = patience  # Early stopping patience
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.scaler = StandardScaler()

    def pretrain_vae(self, dataloader, optimizer):
        """Pre-train the VAE without clustering."""
        self.model.train()
        for epoch in range(self.pretrain_epochs):
            epoch_loss = 0
            for batch in dataloader:
                batch_x = batch[0].to(self.device)
                optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(batch_x)
                loss = self.model.loss_function(reconstructed, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(
                f"Pretrain Epoch {epoch + 1}/{self.pretrain_epochs}, VAE Loss: {epoch_loss / len(dataloader):.4f}"
            )

    def fit(self, X, y=None):
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X.shape[1]
        self.model = VAE(input_dim, self.latent_dim).to(self.device)

        # Optimizer for VAE
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Step 1: Pre-train the VAE
        self.pretrain_vae(dataloader, optimizer)

        # Step 2: Initialize cluster centers with K-means
        print("Initializing cluster centers using K-means...")
        self.model.eval()
        latent_all = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=self.batch_size):
                batch_x = batch[0].to(self.device)
                _, mu, logvar = self.model(batch_x)
                latent_all.append(mu.cpu())  # Use mean (mu) as latent representation
        latent_all = torch.cat(latent_all, dim=0)

        # Perform K-means to initialize cluster centers
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=42)
        kmeans.fit(latent_all.numpy())
        self.cluster_centers = nn.Parameter(
            torch.tensor(
                kmeans.cluster_centers_, dtype=torch.float32, device=self.device
            )
        )

        # Step 3: Jointly train the VAE and clustering objective with regularization and early stopping based on training loss
        optimizer = optim.Adam(
            list(self.model.parameters()) + [self.cluster_centers], lr=self.lr
        )

        best_train_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for batch in dataloader:
                batch_x = batch[0].to(self.device)
                optimizer.zero_grad()
                reconstructed, mu, logvar = self.model(batch_x)

                # Normalize latent vectors and cluster centers dynamically
                latent_norm = nn.functional.normalize(mu, p=2, dim=1)
                centers_norm = nn.functional.normalize(self.cluster_centers, p=2, dim=1)
                # Cosine distance
                cosine_sim = torch.matmul(
                    latent_norm, centers_norm.t()
                )  # (batch_size, n_clusters)
                cosine_distance = 1 - cosine_sim  # (batch_size, n_clusters)
                # Clustering loss: min cosine distance
                clustering_loss = torch.min(cosine_distance, dim=1)[0].mean()

                # Entropy regularization (encourages diversity in cluster assignments)
                soft_assignments = torch.softmax(cosine_sim, dim=1)
                entropy_loss = -torch.mean(
                    torch.sum(
                        soft_assignments * torch.log(soft_assignments + 1e-10), dim=1
                    )
                )

                # Cluster balance regularization (KL divergence with uniform distribution)
                avg_soft_assignments = torch.mean(soft_assignments, dim=0)
                uniform_dist = torch.ones_like(avg_soft_assignments) / self.n_clusters
                kl_divergence_loss = torch.sum(
                    uniform_dist * torch.log(uniform_dist / avg_soft_assignments)
                )

                # VAE loss (reconstruction + KL divergence)
                vae_loss = self.model.loss_function(reconstructed, batch_x, mu, logvar)

                # Total loss: VAE loss + clustering loss + regularization terms
                loss = (
                    vae_loss
                    + self.lambda_ * clustering_loss
                    + self.entropy_lambda * entropy_loss
                    + self.balance_lambda * kl_divergence_loss
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(
                f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {epoch_loss / len(dataloader):.4f}"
            )

            # Early stopping based on training loss
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= self.patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Best training loss: {best_train_loss:.4f}"
                )
                break

        # After training, assign clusters
        self.model.eval()
        latent_all = torch.zeros(len(dataset), self.latent_dim)
        with torch.no_grad():
            for i, batch in enumerate(DataLoader(dataset, batch_size=self.batch_size)):
                batch_x = batch[0].to(self.device)
                _, mu, logvar = self.model(batch_x)
                latent_all[i * self.batch_size : (i + 1) * self.batch_size] = mu.cpu()

            latent_norm = nn.functional.normalize(latent_all, p=2, dim=1)
            centers_norm = nn.functional.normalize(
                self.cluster_centers.data.cpu(), p=2, dim=1
            )
            cosine_sim = torch.matmul(latent_norm, centers_norm.t())  # (N, n_clusters)
            self.labels_ = torch.argmax(cosine_sim, dim=1).numpy()
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Batched prediction to avoid memory overload
        self.model.eval()
        latent_all = []
        with torch.no_grad():
            for batch_x in DataLoader(X_tensor, batch_size=self.batch_size):
                batch_x = batch_x.to(self.device)
                _, mu, logvar = self.model(batch_x)
                latent_all.append(mu.cpu())  # Use mean (mu) for prediction

        latent_all = torch.cat(latent_all, dim=0)
        latent_norm = nn.functional.normalize(latent_all, p=2, dim=1)
        centers_norm = nn.functional.normalize(
            self.cluster_centers.data.cpu(), p=2, dim=1
        )
        cosine_sim = torch.matmul(latent_norm, centers_norm.t())  # (N, n_clusters)
        labels = torch.argmax(cosine_sim, dim=1).cpu().numpy()
        return labels


def create_circle_dataset(
    n_samples=500, n_clusters=4, n_features=2, noise_std=0.05, random_state=42
):
    """
    Creates a circular dataset where the labels are assigned based on the angle.

    Parameters:
    - n_samples (int): Total number of samples.
    - n_clusters (int): Number of clusters (label divisions based on angle).
    - n_features (int): Total number of features (dimensions). If greater than 2, additional random features are added.
    - noise_std (float): Standard deviation of noise to apply to the points on the circle.
    - random_state (int): Random state for reproducibility.

    Returns:
    - X (ndarray): The generated high-dimensional dataset.
    - labels (ndarray): Labels based on the angle.
    """
    np.random.seed(random_state)

    # Create angles for the circle
    theta = np.linspace(
        0, 2 * np.pi, n_samples, endpoint=False
    )  # Ensure no point equals 2*pi

    # Generate circular data (radius = 1) with noise
    r = np.ones_like(theta) + np.random.normal(
        0, noise_std, n_samples
    )  # Constant radius with some noise
    X_circle = np.c_[r * np.cos(theta), r * np.sin(theta)]  # Cartesian coordinates

    # Assign labels based on evenly spaced angle ranges
    angle_bins = np.linspace(0, 2 * np.pi, n_clusters + 1)  # Balanced angle ranges
    labels = (
        np.digitize(theta, bins=angle_bins, right=False) - 1
    )  # Subtract 1 to start labels from 0

    # If n_features > 2, extend the data to higher dimensions
    if n_features > 2:
        additional_features = np.random.randn(
            n_samples, n_features - 2
        )  # Add random features
        X_circle = np.hstack([X_circle, additional_features])

    # Normalize if needed (optional)
    X_circle = normalize(X_circle)

    return X_circle, labels


def visualize_clusters_2d(X, labels, title="Cluster Visualization"):
    """Visualize clusters using the first two dimensions"""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=10)
    plt.title(title)
    plt.xlabel("First Dimension")
    plt.ylabel("Second Dimension")
    plt.colorbar(label="Cluster")
    plt.show()


if __name__ == "__main__":
    # Parameters for dataset
    n_samples = 10000
    n_clusters = 5
    n_features = 3  # Extend to high dimensions

    # Generate the high-dimensional circular dataset using the previous function
    X, y = create_circle_dataset(
        n_samples=n_samples, n_clusters=n_clusters, n_features=n_features
    )
    X = normalize(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Traditional k-means using cosine distance (spherical k-means)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_train)

    # Predict labels for the test set using the fitted k-means model
    kmeans_labels_test = kmeans.predict(X_test)

    # Cosine distance-based k-means accuracy on test set
    kmeans_nmi = normalized_mutual_info_score(y_test, kmeans_labels_test)
    print(f"K-means with Cosine Distance NMI on Test Data: {kmeans_nmi:.4f}")

    # Visualize K-means clustering results using the first two dimensions
    visualize_clusters_2d(
        X_test,
        kmeans_labels_test,
        title="K-means Clustering Results (First 2 Dimensions)",
    )

    # -------------------------------------------
    # Hierarchical Clustering using cosine distance
    # -------------------------------------------

    # Define Agglomerative Clustering with cosine distance
    hierarchical_clustering = AgglomerativeClustering(
        n_clusters=n_clusters, affinity="cosine", linkage="complete"
    )

    # Fit the model on the test data
    hierarchical_labels = hierarchical_clustering.fit_predict(X_test)

    # Evaluate the performance using NMI
    hierarchical_nmi = normalized_mutual_info_score(y_test, hierarchical_labels)
    print(
        f"Hierarchical Clustering (Cosine Distance) NMI on Test Data: {hierarchical_nmi:.4f}"
    )

    # Visualize Hierarchical Clustering results using the first two dimensions
    visualize_clusters_2d(
        X_test,
        hierarchical_labels,
        title="Hierarchical Clustering Results (First 2 Dimensions)",
    )

    # Now, let's apply Deep Clustering with the Autoencoder defined earlier

    # Define deep clustering
    deep_clustering = DeepClustering(
        n_clusters=n_clusters, latent_dim=10, epochs=50, lr=1e-3, lambda_=0.001
    )

    # Fit the deep clustering model on training data
    deep_clustering.fit(X_train)

    # Predict cluster labels on test data
    deep_clustering_labels = deep_clustering.predict(X_test)

    # Evaluate performance using Normalized Mutual Information (NMI) on test data
    deep_clustering_nmi = normalized_mutual_info_score(y_test, deep_clustering_labels)
    print(f"Deep Clustering NMI on Test Data: {deep_clustering_nmi:.4f}")

    # Visualize Deep Clustering results
    visualize_clusters_2d(
        X_test, deep_clustering_labels, title="Deep Clustering Results"
    )
