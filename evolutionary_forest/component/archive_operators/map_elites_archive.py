from deap.tools import HallOfFame
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize


class ACMAPElitesHOF(HallOfFame):
    def __init__(self, maxsize):
        super().__init__(maxsize)

    def update(self, population):
        semantics = [ind.predicted_values for ind in population] + [
            ind.predicted_values for ind in self.items
        ]

        semantics_normalized = normalize(semantics, norm="l2")

        clustering = AgglomerativeClustering(
            n_clusters=self.maxsize, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(semantics_normalized)

        cluster_individuals = {i: [] for i in range(self.maxsize)}
        for label, ind in zip(labels, population + list(self.items)):
            cluster_individuals[label].append(ind)

        new_hof = []
        for cluster_inds in cluster_individuals.values():
            if cluster_inds:
                best_individual = min(cluster_inds, key=lambda ind: ind.fitness.values)
                new_hof.append(best_individual)

        self.clear()
        super().update(new_hof)
