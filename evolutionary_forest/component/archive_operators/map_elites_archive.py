from deap.tools import HallOfFame, selBest
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


class ACMAPElitesHOF(HallOfFame):
    def __init__(self, maxsize, map_archive_candidate_size=3):
        super().__init__(maxsize)
        self.map_archive_candidate_size = map_archive_candidate_size

    def update(self, population):
        best_candidate = selBest(
            population + list(self.items),
            self.maxsize * self.map_archive_candidate_size,
        )
        semantics = [ind.predicted_values for ind in best_candidate]

        semantics_normalized = normalize(semantics, norm="l2")

        clustering = AgglomerativeClustering(
            n_clusters=self.maxsize, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(semantics_normalized)

        cluster_individuals = {i: [] for i in range(self.maxsize)}
        for label, ind in zip(labels, best_candidate):
            cluster_individuals[label].append(ind)

        new_hof = []
        for cluster_inds in cluster_individuals.values():
            if cluster_inds:
                best_individual = max(cluster_inds, key=lambda ind: ind.fitness.wvalues)
                new_hof.append(best_individual)

        self.clear()
        super().update(new_hof)
