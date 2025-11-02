import numpy as np
from deap.tools import HallOfFame, selBest
from sklearn.metrics.pairwise import cosine_similarity


class DPPEnsembleHOF(HallOfFame):
    def __init__(self, maxsize, y, verbose=False):
        super().__init__(maxsize)
        self.y = y
        assert isinstance(self.y, np.ndarray)
        self.verbose = verbose

    def update(self, population):
        candidates = selBest(population, self.maxsize) + list(self.items)
        X = (
            np.array([ind.predicted_values for ind in candidates]) - self.y
        )  # centered by y
        q = np.array([ind.fitness.wvalues[0] for ind in candidates])
        q = (q - q.min()) / (q.max() - q.min() + 1e-12)
        L = np.outer(q, q) * cosine_similarity(X)

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
                # mask already selected indices to prevent repeats
                scores[selected] = -np.inf
                i = np.argmax(scores)
            selected.append(i)

        new_hof = [candidates[i] for i in selected]
        if self.verbose:
            print(f"[DPP] Selected {len(new_hof)} diverse high-fitness individuals")
        self.clear()
        super().update(new_hof)
