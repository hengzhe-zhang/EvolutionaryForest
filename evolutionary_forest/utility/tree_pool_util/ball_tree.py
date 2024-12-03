import numpy as np
from sklearn.neighbors import BallTree


class ScipyBallTree(BallTree):
    def query(self, X, k=1, return_distance=True, dualtree=False, breadth_first=False):
        if len(X.shape) == 1:
            distances, indices = super().query(
                X.reshape(1, -1), k, return_distance, dualtree, breadth_first
            )
            if len(distances[0]) == 1:
                # return a single value if k=1
                return distances[0][0], indices[0][0]
            else:
                return distances[0], indices[0]
        else:
            return super().query(X, k, return_distance, dualtree, breadth_first)


if __name__ == "__main__":
    data_points = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])

    tree = ScipyBallTree(data_points)

    query_point = np.array([1.5, 1.5])  # A single query point
    distances, indices = tree.query(query_point, k=2)  # Find the 2 nearest neighbors

    # Display the results
    print("Query Point:", query_point)
    print("Distances to Neighbors:", distances)
    print("Indices of Neighbors:", indices)
