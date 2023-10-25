import random
import numpy as np


class KnEABinaryTournament:
    def __init__(self, population, K, N):
        """
        Initialize the binary tournament selection for KnEA.

        :param population: List of solutions in the population
        :param K: Set of knee points
        :param N: Desired size of the mating pool
        """
        self.population = population
        self.K = K
        self.N = N

    def dominance(self, a, b):
        """
        Check if solution 'a' dominates solution 'b'.

        :param a: Solution a
        :param b: Solution b
        :return: True if a dominates b, otherwise False
        """
        return a < b

    def weighted_distance(self, p, k):
        """
        Calculate the weighted distance of a solution based on its k-nearest neighbors.

        :param p: Solution for which to calculate the weighted distance
        :param k: Number of nearest neighbors to consider
        :return: Weighted distance of the solution p
        """
        distances = [np.linalg.norm(p - neighbor) for neighbor in self.population]
        sorted_neighbors = sorted(
            [(dist, idx) for idx, dist in enumerate(distances)], key=lambda x: x[0]
        )
        nearest_neighbors = sorted_neighbors[:k]
        sum_distance = sum([dist for dist, _ in nearest_neighbors])

        r_values = [
            1 / (abs(dist - (1 / k) * sum_distance)) for dist, _ in nearest_neighbors
        ]
        w_values = [r / sum(r_values) for r in r_values]

        DW = sum([w * dist for w, (dist, _) in zip(w_values, nearest_neighbors)])

        return DW

    def select(self):
        """
        Perform binary tournament selection to select solutions for mating.

        :return: List of selected solutions for mating
        """
        Q = []

        while len(Q) < self.N:
            a, b = random.sample(self.population, 2)

            if self.dominance(a, b):
                Q.append(a)
            elif self.dominance(b, a):
                Q.append(b)
            else:
                if a in self.K and b not in self.K:
                    Q.append(a)
                elif a not in self.K and b in self.K:
                    Q.append(b)
                else:
                    DW_a = self.weighted_distance(a, len(self.population))
                    DW_b = self.weighted_distance(b, len(self.population))

                    if DW_a > DW_b:
                        Q.append(a)
                    elif DW_a < DW_b:
                        Q.append(b)
                    else:
                        Q.append(a if random.random() < 0.5 else b)

        return Q
