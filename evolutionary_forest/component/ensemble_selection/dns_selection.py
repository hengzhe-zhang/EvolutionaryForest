import numpy as np
from deap.tools import HallOfFame, selBest


def dns_ensemble_selection(pop, y, k=5, top_n=10):
    """
    Dominated Novelty Search ensemble selection for evolutionary populations.
    Uses cosine-style distances after centering predictions by y.

    Args:
        pop : list
            Population where each individual has .semantics and .fitness.wvalues[0].
        y : np.ndarray, shape (n_samples,)
            True targets for centering.
        k : int
            Number of nearest fitter individuals to consider.
        top_n : int
            Number of individuals to select.
    Returns:
        selected_idx : np.ndarray of int
            Indices of selected individuals.
    """
    preds = np.array([ind.predicted_values for ind in pop])
    fitness = np.array([ind.fitness.wvalues[0] for ind in pop])

    # Center by true y and normalize (for cosine-like distance)
    centered = preds - y
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-12
    normed = centered / norms

    n = len(fitness)
    dns_scores = np.zeros(n)

    for i in range(n):
        fitter = np.where(fitness > fitness[i])[0]
        if len(fitter) == 0:
            dns_scores[i] = np.inf
            continue
        # Cosine distance = 1 - cosine similarity
        cos_sim = normed[fitter] @ normed[i]
        dists = 1 - cos_sim
        dns_scores[i] = np.mean(np.sort(dists)[: min(k, len(dists))])

    selected_idx = np.argsort(-dns_scores)[:top_n]
    return selected_idx


class DNSHOF(HallOfFame):
    def __init__(
        self, maxsize, y, map_elites_hof_mode="Independent", k=5, verbose=False
    ):
        """
        Hall of Fame using Dominated Novelty Search (DNS) ensemble selection.

        Args:
            maxsize : int
                Maximum size of the Hall of Fame.
            y : np.ndarray
                True target values for centering semantics.
            map_elites_hof_mode : str
                "Free" or "Independent" mode for candidate pool construction.
            k : int
                Number of nearest fitter individuals for DNS competition.
            verbose : bool
                If True, print fitness stats after update.
        """
        super().__init__(maxsize)
        self.y = y
        assert isinstance(self.y, np.ndarray)
        self.map_elites_hof_mode = map_elites_hof_mode
        self.k = k
        self.verbose = verbose

    def update(self, population):
        # --- Construct candidate pool ---
        if self.map_elites_hof_mode == "Free":
            candidates = population + list(self.items)
        elif self.map_elites_hof_mode == "Independent":
            candidates = selBest(population, self.maxsize) + list(self.items)
        else:
            raise ValueError("map_elites_hof_mode must be 'Free' or 'Independent'.")

        if len(candidates) == 0:
            return

        # --- DNS selection ---
        selected_idx = dns_ensemble_selection(
            pop=candidates, y=self.y, k=self.k, top_n=min(self.maxsize, len(candidates))
        )
        new_hof = [candidates[i] for i in selected_idx]

        # --- Verbose summary ---
        if self.verbose:
            hof_fit = [ind.fitness.wvalues[0] for ind in new_hof]
            pop_fit = [ind.fitness.wvalues[0] for ind in population]
            print(
                f"[DNS-HOF] mean={np.mean(hof_fit):.3f}, std={np.std(hof_fit):.3f}, "
                f"min={np.min(hof_fit):.3f}, max={np.max(hof_fit):.3f}"
            )
            print(
                f"[Population] mean={np.mean(pop_fit):.3f}, std={np.std(pop_fit):.3f}, "
                f"min={np.min(pop_fit):.3f}, max={np.max(pop_fit):.3f}"
            )

        # --- Update Hall of Fame ---
        self.clear()
        super().update(new_hof)
