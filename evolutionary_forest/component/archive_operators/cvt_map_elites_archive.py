import numpy as np
from deap.tools import HallOfFame, selBest
from sklearn.cluster import (
    AgglomerativeClustering,
    SpectralClustering,
    KMeans,
)

from evolutionary_forest.model.cosine_kmeans import CosineKMeans


def visualize_kmeans_clustering_separately(
    semantics_1,
    semantics_2,
    fitness,
    target,
    n_clusters=3,
    use_tsne=False,
    use_first_two_dims=False,
):
    """
    Visualize K-Means clustering results separately for two semantics arrays using PCA or t-SNE.

    Parameters:
    semantics_1 (np.array): First semantics array.
    semantics_2 (np.array): Second semantics array.
    n_clusters (int): Number of clusters for K-Means.
    use_tsne (bool): Whether to use t-SNE for visualization. If False, PCA will be used.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    def plot_clustering(semantics, title):
        # Apply K-Means clustering
        kmeans = CosineKMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(semantics)

        # Dimensionality reduction
        if use_first_two_dims:
            reduced_data = semantics[:, :2]
        elif use_tsne:
            reducer = TSNE(n_components=2, random_state=0)
            reduced_data = reducer.fit_transform(semantics)
        else:
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(semantics)

        # Plot the clustering results
        plt.figure(figsize=(10, 6))
        for i in range(n_clusters):
            cluster_points = reduced_data[labels == i]
            if len(cluster_points) == 0:
                continue
            cluster_fitness = [
                fitness[j] for j in range(len(fitness)) if labels[j] == i
            ]
            max_fitness_index = np.argmax(cluster_fitness)
            plt.scatter(
                cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}"
            )
            # Highlight the point with the largest fitness in the cluster
            plt.scatter(
                cluster_points[max_fitness_index, 0],
                cluster_points[max_fitness_index, 1],
                color="red",
                s=100,
                edgecolor="k",
                label=f"Cluster {i} Max Fitness",
            )

        # Plot the target values
        plt.scatter(
            target[0],
            target[1],
            cmap="viridis",
            marker="x",
            s=300,
            label="Target",
        )

        # Annotate fitness values
        for i, (x, y) in enumerate(reduced_data):
            plt.text(x, y, f"{fitness[i]:.2f}", fontsize=8, ha="right")

        plt.title(title)
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.colorbar(label="Target Value")
        plt.show()

    plot_clustering(
        semantics_1,
        "K-Means Clustering Visualization for Semantics 1 ({}-D)".format(
            "t-SNE" if use_tsne else "PCA"
        ),
    )
    plot_clustering(
        semantics_2,
        "K-Means Clustering Visualization for Semantics 2 ({}-D)".format(
            "t-SNE" if use_tsne else "PCA"
        ),
    )


class CVTMAPElitesHOF(HallOfFame):
    def __init__(
        self,
        maxsize,
        map_archive_candidate_size=None,
        clustering_method="KMeans-Cosine",
        map_elites_hof_mode="Independent",
        y=None,
        symmetric_map_archive_mode=False,
        centered_cvt_map_elites=True,
        **kwargs,
    ):
        super().__init__(maxsize)
        self.symmetric_map_archive_mode = symmetric_map_archive_mode
        self.map_archive_candidate_size = map_archive_candidate_size
        if self.map_archive_candidate_size is None:
            self.map_archive_candidate_size = maxsize
        self.clustering_method = clustering_method
        self.map_elites_hof_mode = map_elites_hof_mode
        self.centered_cvt_map_elites = centered_cvt_map_elites
        self.y = y
        assert isinstance(self.y, np.ndarray)

    def update(self, population):
        if self.map_elites_hof_mode == "Independent":
            best_candidate = selBest(
                population,
                self.map_archive_candidate_size,
            ) + list(self.items)
        else:
            best_candidate = selBest(
                population + list(self.items),
                self.map_archive_candidate_size,
            )
        # centered
        if self.centered_cvt_map_elites:
            """
            Must be centered, otherwise the center is not the target semantics
            """
            semantics = np.array(
                [ind.predicted_values + self.y for ind in best_candidate]
            )
        else:
            semantics = np.array([ind.predicted_values for ind in best_candidate])

        # semantics_a = np.array([ind.predicted_values for ind in best_candidate])
        # semantics_a = semantics_a + self.y
        #
        # semantics_b = np.array([ind.predicted_values for ind in best_candidate])
        # visualize_kmeans_clustering_separately(
        #     semantics_a[:, :2],
        #     semantics_b[:, :2],
        #     fitness=[ind.fitness.wvalues[0] for ind in best_candidate],
        #     target=self.y[:2],
        #     n_clusters=5,
        #     use_first_two_dims=True,
        # )

        if self.symmetric_map_archive_mode:
            symmetric_semantics = -semantics
            semantics = np.concatenate([semantics, symmetric_semantics], axis=0)

        if self.clustering_method.startswith("Agglomerative"):
            _, metric, linkage = self.clustering_method.split("-")
            metric = metric.lower()
            linkage = linkage.lower()
            clustering = AgglomerativeClustering(
                n_clusters=self.maxsize, metric=metric, linkage=linkage
            )
        # elif self.clustering_method == "KMeans-Cosine":
        #     semantics = normalize(semantics, norm="l2")
        #     clustering = KMeans(n_clusters=self.maxsize, random_state=0)
        elif self.clustering_method == "KMeans-Cosine":
            clustering = CosineKMeans(n_clusters=self.maxsize, random_state=0)
        elif self.clustering_method == "KMeans":
            clustering = KMeans(n_clusters=self.maxsize, random_state=0)
        elif self.clustering_method == "Spectral":
            clustering = SpectralClustering(
                n_clusters=self.maxsize, affinity="nearest_neighbors"
            )
        else:
            raise ValueError("Unsupported clustering method")

        labels = clustering.fit_predict(semantics)

        cluster_individuals = {i: [] for i in range(self.maxsize)}
        original_length = len(best_candidate)
        for label, ind in zip(labels[:original_length], best_candidate):
            cluster_individuals[label].append(ind)

        new_hof = []
        for cluster_inds in cluster_individuals.values():
            if cluster_inds:
                best_individual = max(cluster_inds, key=lambda ind: ind.fitness.wvalues)
                new_hof.append(best_individual)

        self.clear()
        super().update(new_hof)
