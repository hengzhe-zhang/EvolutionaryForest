import os

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances


def knn_neighbor_target_variance(features, labels, k_neighbors):
    """
    For each point i, k nearest neighbors in the given coordinate space (rows of
    ``features``), then mean of Var(labels[neighbors]) over i.
    """
    dist = pairwise_distances(features, metric="euclidean")
    nn_idx = np.argsort(dist, axis=1)[:, 1 : k_neighbors + 1]
    nn_targets = labels[nn_idx]
    variances = np.var(nn_targets, axis=1, ddof=0)
    return float(np.mean(variances))


def pca_knn_target_variance(transformer_feature, y, k_neighbors, n_components=2):
    """Same PCA as ``pca_plot``; variance is over neighbors in PCA space."""
    pca = PCA(n_components=n_components)
    space = pca.fit_transform(transformer_feature)
    return knn_neighbor_target_variance(space, y, k_neighbors)


def pca_plot(
    transformer_feature,
    y,
    figname,
    n_components=2,
    subtitle=None,
    knn_target_var_k=None,
):
    pca = PCA(n_components=n_components)
    space = pca.fit_transform(transformer_feature)
    if knn_target_var_k is not None:
        v = knn_neighbor_target_variance(space, y, knn_target_var_k)
        subtitle = f"{knn_target_var_k}-NN Target Variance: {v:.3g}"
    make_plot(space, y, figname, subtitle=subtitle)


def tsne_plot(transformer_feature, y, figname, n_components=2):
    # Apply PCA to reduce dimensions
    pca = TSNE(n_components=n_components)
    space = pca.fit_transform(transformer_feature)
    make_plot(space, y, figname)


def make_plot(x_space, y, figname, subtitle=None):
    # Normalize the continuous values for color mapping
    norm = mcolors.Normalize(vmin=min(y), vmax=max(y))
    cmap = cm.viridis

    # Create the scatter plot
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    ax = plt.gca()
    scatter = ax.scatter(
        x_space[:, 0], x_space[:, 1], c=y, cmap=cmap, norm=norm, alpha=0.7
    )

    # Add a colorbar to show the scale of the continuous values
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Continuous Value")

    # Set plot titles and labels (title matches axis label size)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    label_fs = ax.xaxis.get_label().get_fontsize()
    if subtitle:
        ax.set_title(subtitle, fontsize=label_fs, pad=6)
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

    get_rank = False
    if get_rank:
        # Rank distances (lower rank = closer neighbor)
        dist_original = np.argsort(np.argsort(dist_original, axis=1), axis=1)
        dist_constructed = np.argsort(np.argsort(dist_constructed, axis=1), axis=1)
        dist_transformed = np.argsort(np.argsort(dist_transformed, axis=1), axis=1)
        dist_y = np.argsort(np.argsort(dist_y, axis=1), axis=1)

        print(
            "Squared error of distances:",
            np.mean((dist_original - dist_y) ** 2),
            np.mean((dist_constructed - dist_y) ** 2),
            np.mean((dist_transformed - dist_y) ** 2),
        )

    # Use a shared color scale across all four heatmaps.
    # This keeps the colorbar comparable when we visualize raw distances.
    vmin = float(
        min(
            np.min(dist_original),
            np.min(dist_constructed),
            np.min(dist_transformed),
            np.min(dist_y),
        )
    )
    vmax = float(
        max(
            np.max(dist_original),
            np.max(dist_constructed),
            np.max(dist_transformed),
            np.max(dist_y),
        )
    )

    # Plot and save Original Feature Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_original, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "original_feature_distances.eps"), format="eps"
    )

    # Plot and save Constructed Feature Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_constructed, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "constructed_feature_distances.eps"), format="eps"
    )

    # Plot and save Transformed Feature Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_transformed, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "transformed_feature_distances.eps"), format="eps"
    )

    # Plot and save Target Pairwise Distances
    plt.figure(figsize=(8 * 0.5, 6 * 0.5))
    plt.imshow(dist_y, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(
        os.path.join(result_folder, "target_pairwise_distances.eps"), format="eps"
    )


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
