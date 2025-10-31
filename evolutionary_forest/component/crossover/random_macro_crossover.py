from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP, get_semantic_results


def randomClusteringCrossover(
    ind1: MultipleGeneGP, ind2: MultipleGeneGP, target: np.ndarray
):
    _, semantic_result = get_semantic_results(ind1, ind2, target)
    semantic_result = normalize(semantic_result, norm="l2")

    # remove duplicate semantic vectors
    _, unique_idx = np.unique(semantic_result, axis=0, return_index=True)
    semantic_result = semantic_result[unique_idx]

    n_clusters = ind1.gene_num
    if len(semantic_result) < n_clusters:
        return []

    # k-means clustering
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(semantic_result)

    # randomly select one index per cluster
    selected_unique = [
        np.random.choice(np.where(labels == c)[0])
        for c in range(n_clusters)
        if np.any(labels == c)
    ]
    selected = [unique_idx[i] for i in selected_unique]
    if len(selected) != ind1.gene_num:
        return []

    # rebuild genes
    combined_genes = ind1.gene + ind2.gene
    ind1.gene = [combined_genes[i] for i in selected]

    assert len(ind1.gene) == len(ind2.gene)
    return [ind1]
