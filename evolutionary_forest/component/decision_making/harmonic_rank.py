import numpy as np


def best_harmonic_rank(front):
    reciprocals = 1.0 / front
    harmonic_means = front.shape[1] / np.sum(reciprocals, axis=1)
    return np.argmax(harmonic_means)


if __name__ == '__main__':
    # Example usage:
    front = np.array([[2, 4, 6],
                      [1, 3, 9],
                      [4, 5, 3]])

    print(best_harmonic_rank(front))
