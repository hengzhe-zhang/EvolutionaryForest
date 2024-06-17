import numpy as np
from deap.tools import HallOfFame, selBest
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
)
from sklearn.preprocessing import normalize

from evolutionary_forest.model.cosine_kmeans import CosineKMeans


class CVTMAPElitesHOF(HallOfFame):
    def __init__(
        self,
        maxsize,
        map_archive_candidate_size=None,
        clustering_method="KMeans-Cosine",
        map_elites_hof_mode="Independent",
        y=None,
        symmetric_map_archive_mode=False,
        **kwargs
    ):
        super().__init__(maxsize)
        self.symmetric_map_archive_mode = symmetric_map_archive_mode
        self.map_archive_candidate_size = map_archive_candidate_size
        if self.map_archive_candidate_size is None:
            self.map_archive_candidate_size = maxsize
        self.clustering_method = clustering_method
        self.map_elites_hof_mode = map_elites_hof_mode
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
        semantics = np.array([ind.predicted_values - self.y for ind in best_candidate])

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
