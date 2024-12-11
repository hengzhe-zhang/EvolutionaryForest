import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def pca_plot(transformer_feature, y, n_components=2):
    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    space = pca.fit_transform(transformer_feature)

    # Normalize the continuous values for color mapping
    norm = mcolors.Normalize(vmin=min(y), vmax=max(y))
    cmap = cm.viridis

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        space[:, 0], space[:, 1], c=y, cmap=cmap, norm=norm, alpha=0.7
    )

    # Add a colorbar to show the scale of the continuous values
    cbar = plt.colorbar(scatter)
    cbar.set_label("Continuous Value")

    # Set plot titles and labels
    plt.title("PCA Visualization of Features (Continuous y)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.show()


def plot_pairwise_distances(original_feature, transformed_feature, y):
    # Calculate pairwise distances
    dist_original = pairwise_distances(original_feature, metric="euclidean")
    dist_transformed = pairwise_distances(transformed_feature, metric="euclidean")
    dist_y = pairwise_distances(y.reshape(-1, 1), metric="euclidean")

    # Plot pairwise distances
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Feature Distances
    im0 = axes[0].imshow(dist_original, aspect="auto", cmap="viridis")
    axes[0].set_title("Original Feature Pairwise Distances")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Sample Index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Transformed Feature Distances
    im1 = axes[1].imshow(dist_transformed, aspect="auto", cmap="viridis")
    axes[1].set_title("Transformed Feature Pairwise Distances")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Sample Index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Target Feature Distances
    im2 = axes[2].imshow(dist_y, aspect="auto", cmap="viridis")
    axes[2].set_title("Target Feature Pairwise Distances")
    axes[2].set_xlabel("Sample Index")
    axes[2].set_ylabel("Sample Index")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def pairwise_distance_plot():
    original_feature = np.random.rand(100, 5)
    transformed_feature = np.random.rand(100, 5)
    weight = np.random.rand(5, 5)
    y = np.random.rand(100, 5)
    plot_pairwise_distances(original_feature, transformed_feature, weight, y)


if __name__ == "__main__":
    # pairwise_distance_plot()
    transformer_feature = np.random.rand(100, 50)  # 100 samples, 50 features
    y = np.random.rand(100)  # Continuous labels (e.g., values between 0 and 1)

    # Plot t-SNE visualization
    pca_plot(transformer_feature, y)
