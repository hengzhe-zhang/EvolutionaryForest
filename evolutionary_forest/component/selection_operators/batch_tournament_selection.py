import random
from typing import Sequence, List, Optional, Callable
import numpy as np


def selBatchTournament(
    individuals: Sequence,
    k: int,
    *,
    batch_size: int = 16,
    tourn_size: int = 16,
    shuffle: bool = True,
    get_errors: Optional[Callable] = None,
    rng: random.Random = random,
) -> List:
    """
    NumPy-only Batch Tournament Selection (BTS/BTSS).

    - Uses mean error on each batch as the tournament score.
    - If shuffle=False (BTS): order batches by decreasing difficulty
      measured on the current best individual (hardest cases first).
    - If shuffle=True (BTSS): randomly permute cases each outer round.
    """
    if k <= 0:
        return []
    if not individuals:
        raise ValueError("Population is empty.")
    if get_errors is None:
        get_errors = lambda ind: getattr(ind, "semantics")

    # Error matrix: shape (N_individuals, N_cases)
    E = np.asarray([get_errors(ind) for ind in individuals], dtype=float)
    if E.ndim != 2 or E.shape[1] == 0:
        raise ValueError(
            "Each individual's error vector must be non-empty and same length."
        )
    n_ind, n_cases = E.shape

    # Best individual by overall mean error
    means = E.mean(axis=1)
    best_idx = int(np.argmin(means))
    best_err = E[best_idx]

    # Initial case order
    if shuffle:
        case_idx = np.arange(n_cases)
        rng.shuffle(case_idx)
    else:
        # Hardest first: sort by descending error on the best individual
        case_idx = np.argsort(-best_err, kind="mergesort")  # stable for determinism

    # Split into batches
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1.")
    batches = [case_idx[i : i + batch_size] for i in range(0, n_cases, batch_size)]

    indices = np.arange(n_ind)
    selected = []

    while len(selected) < k:
        # BTSS: reshuffle cases each outer round
        if shuffle:
            case_idx = np.arange(n_cases)
            rng.shuffle(case_idx)
            batches = [
                case_idx[i : i + batch_size] for i in range(0, n_cases, batch_size)
            ]

        for b in batches:
            if len(selected) >= k:
                break

            # Draw tournament
            if tourn_size <= 0:
                raise ValueError("tourn_size must be >= 1.")
            if tourn_size <= n_ind:
                cand_idx = rng.sample(range(n_ind), tourn_size)
            else:
                cand_idx = [rng.randrange(n_ind) for _ in range(tourn_size)]

            # Mean error on the current batch
            cand_errors = E[cand_idx][:, b]  # (tourn_size, |b|)
            cand_means = cand_errors.mean(axis=1)  # (tourn_size,)
            winner_local = int(np.argmin(cand_means))
            winner_idx = cand_idx[winner_local]
            selected.append(individuals[winner_idx])

    return selected
