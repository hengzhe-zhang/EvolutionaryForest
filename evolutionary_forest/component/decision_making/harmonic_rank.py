import numpy as np


def best_harmonic_rank(front):
    # Rank within each column
    ranks = front.argsort(axis=0).argsort(axis=0) + 1
    reciprocals_of_ranks = 1.0 / ranks
    harmonic_means_per_column = front.shape[1] / np.sum(reciprocals_of_ranks, axis=1)
    return np.argmax(harmonic_means_per_column)


if __name__ == '__main__':
    # Example usage:
    front = np.array([[2, 4],
                      [1, 3],
                      [4, 5]])

    print(best_harmonic_rank(front))
