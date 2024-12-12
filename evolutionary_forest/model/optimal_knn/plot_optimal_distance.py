import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


def pca_plot(transformer_feature, y, figname, n_components=2):
    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    space = pca.fit_transform(transformer_feature)

    # Normalize the continuous values for color mapping
    norm = mcolors.Normalize(vmin=min(y), vmax=max(y))
    cmap = cm.viridis

    # Create the scatter plot
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    scatter = plt.scatter(
        space[:, 0], space[:, 1], c=y, cmap=cmap, norm=norm, alpha=0.7
    )

    # Add a colorbar to show the scale of the continuous values
    cbar = plt.colorbar(scatter)
    cbar.set_label("Continuous Value")

    # Set plot titles and labels
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join("result", figname), format="eps")
    plt.show()


def plot_pairwise_distances(
    original_feature,
    constructed_feature,
    transformed_feature,
    y,
    result_folder="result",
):
    # Calculate pairwise distances
    dist_original = pairwise_distances(original_feature, metric="euclidean")
    dist_constructed = pairwise_distances(constructed_feature, metric="euclidean")
    dist_transformed = pairwise_distances(transformed_feature, metric="euclidean")
    dist_y = pairwise_distances(y.reshape(-1, 1), metric="euclidean")

    # Plot and save Original Feature Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_original, aspect="auto", cmap="viridis")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "original_feature_distances.eps"), format="eps"
    )
    plt.close()

    # Plot and save Constructed Feature Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_constructed, aspect="auto", cmap="viridis")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "constructed_feature_distances.eps"), format="eps"
    )
    plt.close()

    # Plot and save Transformed Feature Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_transformed, aspect="auto", cmap="viridis")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "transformed_feature_distances.eps"), format="eps"
    )
    plt.close()

    # Plot and save Target Pairwise Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_y, aspect="auto", cmap="viridis")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "target_pairwise_distances.eps"), format="eps"
    )
    plt.close()


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
