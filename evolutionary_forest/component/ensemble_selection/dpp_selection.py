import numpy as np
from deap.tools import HallOfFame, selBest
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def compute_similarity(X, metric="cosine", gamma=None):
    if metric == "cosine":
        return cosine_similarity(X)
    elif metric == "rbf":
        D2 = euclidean_distances(X, squared=True)
        if gamma is None:
            # ICML-style median heuristic for gamma
            gamma = 1 / (2 * np.median(D2[D2 > 0]))  # avoid zeros on diagonal
        return np.exp(-gamma * D2)
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")


class DPPEnsembleHOF(HallOfFame):
    def __init__(self, maxsize, y, metric="cosine", gamma=None, verbose=False):
        super().__init__(maxsize)
        assert isinstance(y, np.ndarray)
        self.y = y
        self.metric = metric
        self.gamma = gamma
        self.verbose = verbose

    def update(self, population):
        candidates = selBest(population, self.maxsize) + list(self.items)
        X = np.array([ind.predicted_values for ind in candidates]) - self.y

        q = np.array([ind.fitness.wvalues[0] for ind in candidates])
        q = (q - q.min()) / (q.max() - q.min() + 1e-12)

        S = compute_similarity(X, metric=self.metric, gamma=self.gamma)
        L = np.outer(q, q) * S

        selected = []
        for _ in range(self.maxsize):
            if not selected:
                i = np.argmax(np.diag(L))
            else:
                Li = L[np.ix_(selected, selected)]
                vi = L[:, selected]
                scores = np.diag(L) - np.sum(
                    (vi @ np.linalg.pinv(Li + 1e-8 * np.eye(len(Li)))) * vi, axis=1
                )
                scores[selected] = -np.inf
                i = np.argmax(scores)
            selected.append(i)

        self.clear()
        super().update([candidates[i] for i in selected])
        if self.verbose:
            print(
                f"[DPP] Selected {len(self)} individuals using {self.metric} similarity"
            )
