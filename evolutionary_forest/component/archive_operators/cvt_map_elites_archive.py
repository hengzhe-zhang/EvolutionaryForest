import numpy as np
from deap.tools import HallOfFame, selBest
from sklearn.cluster import (
    AgglomerativeClustering,
    SpectralClustering,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from evolutionary_forest.component.deep_clustering.vae_clustering import DeepClustering
from evolutionary_forest.model.clustering.shapley_pruning import (
    prune_models_based_on_shapley_for_regression,
)
from evolutionary_forest.model.cosine_kmeans import (
    CosineKMeans,
    CosineKMedoids,
    select_medoid,
)


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
        deep_clustering_parameters=None,
        verbose=False,
        adaptive_switch_map_elites=None,
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

        if self.clustering_method.startswith("Agglomerative"):
            _, metric, linkage = self.clustering_method.split("-")
            metric = metric.lower()
            linkage = linkage.lower()
            clustering = AgglomerativeClustering(
                n_clusters=self.maxsize, metric=metric, linkage=linkage
            )
        elif (
            self.clustering_method == "KMeans-Cosine"
            or self.clustering_method == "Shapley-KMeans-Cosine"
        ):
            clustering = CosineKMeans(n_clusters=self.maxsize, random_state=0)
        elif (
            self.clustering_method == "KMedoids-Cosine"
            or self.clustering_method == "KMedoids-Cosine+"
        ):
            clustering = CosineKMedoids(n_clusters=self.maxsize, random_state=0)
        elif self.clustering_method == "KMeans":
            clustering = KMeans(n_clusters=self.maxsize, random_state=0)
        elif self.clustering_method == "DeepClustering":
            clustering = DeepClustering(
                n_clusters=self.maxsize,
                **deep_clustering_parameters,
            )
        elif self.clustering_method == "Spectral":
            clustering = SpectralClustering(
                n_clusters=self.maxsize, affinity="nearest_neighbors"
            )
        else:
            raise ValueError("Unsupported clustering method")

        self.clustering = clustering
        self.verbose = verbose
        self.adaptive_switch_map_elites = adaptive_switch_map_elites

    def update(self, population):
        if self.map_elites_hof_mode == "Free":
            best_candidate = population + list(self.items)
        elif self.map_elites_hof_mode == "Independent":
            best_candidate = selBest(
                population,
                self.map_archive_candidate_size,
            ) + list(self.items)
        elif self.map_elites_hof_mode == "Combined":
            best_candidate = selBest(
                population + list(self.items),
                self.map_archive_candidate_size,
            )
        else:
            raise ValueError("Unsupported map elites hof mode")
        if len(best_candidate) < self.maxsize:
            self.clear()
            super().update(best_candidate)
            return
        # centered
        semantics = np.array([ind.predicted_values for ind in best_candidate]) - self.y

        if self.symmetric_map_archive_mode:
            symmetric_semantics = -semantics
            semantics = np.concatenate([semantics, symmetric_semantics], axis=0)

        if isinstance(self.clustering, DeepClustering):
            semantics = StandardScaler().fit_transform(semantics)
            self.clustering.fit(semantics)
            labels = self.clustering.predict(semantics)
        else:
            labels = self.clustering.fit_predict(semantics)

        cluster_individuals = {i: [] for i in range(self.maxsize)}
        original_length = len(best_candidate)
        for label, ind in zip(labels[:original_length], best_candidate):
            cluster_individuals[label].append(ind)

        new_hof = []
        for cluster_idx, cluster_inds in cluster_individuals.items():
            if cluster_inds:
                if self.clustering_method == "KMedoids-Cosine+":
                    # Get the corresponding semantics for this cluster
                    cluster_semantics = semantics[labels == cluster_idx]

                    # Find the medoid by minimizing the sum of distances within the cluster
                    medoid_idx = select_medoid(cluster_semantics)
                    new_hof.append(cluster_inds[medoid_idx])
                else:
                    best_individual = max(
                        cluster_inds, key=lambda ind: ind.fitness.wvalues
                    )
                    new_hof.append(best_individual)

        if self.clustering_method.startswith("Shapley"):
            index = prune_models_based_on_shapley_for_regression(
                np.array([ind.predicted_values for ind in new_hof]), self.y
            )
            new_hof = [new_hof[i] for i in index]

        if self.verbose:
            hof_std = np.std([ind.fitness.wvalues[0] for ind in new_hof])
            hof_mean = np.mean([ind.fitness.wvalues[0] for ind in new_hof])
            hof_min = np.min([ind.fitness.wvalues[0] for ind in new_hof])
            print(f"Mean={hof_mean:.3f}, Std={hof_std:.3f}, Min={hof_min:.3f}")
            pop_std = np.std([ind.fitness.wvalues[0] for ind in population])
            pop_mean = np.mean([ind.fitness.wvalues[0] for ind in population])
            pop_min = np.min([ind.fitness.wvalues[0] for ind in population])
            print(f"Mean={pop_mean:.3f}, Std={pop_std:.3f}, Min={pop_min:.3f}")

        std_r2 = np.std([ind.fitness.wvalues[0] for ind in new_hof])
        if self.adaptive_switch_map_elites == "MAP-STD" and std_r2 <= 0.02:
            if self.verbose:
                print(
                    f"[Adaptive switch] Std(R²)={std_r2:.3f} < 0.02 → Using Top selection instead."
                )
            new_hof = selBest(population + list(self.items), self.maxsize)

        self.clear()
        super().update(new_hof)
