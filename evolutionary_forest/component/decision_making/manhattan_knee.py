import numpy as np
from matplotlib import pyplot as plt


def find_manhattan_knee(pareto_points):
    """
    Identify the knee point based on the bend angle.
    Returns both the knee point and its index.
    """
    distance_function = lambda p1, p2: np.max(p1 - p2)

    # Sort pareto_points by the first objective in descending order while keeping track of original indices
    sorted_points_with_indices = sorted(
        enumerate(pareto_points), key=lambda x: x[1][0], reverse=False
    )
    sorted_indices = [x[0] for x in sorted_points_with_indices]
    sorted_points = [x[1] for x in sorted_points_with_indices]

    if len(pareto_points) < 3:
        return sorted_points[0], sorted_indices[0]

    # min-max normalization
    sorted_points = np.array(sorted_points)
    sorted_points = (sorted_points - np.min(sorted_points, axis=0)) / (
        np.max(sorted_points, axis=0) - np.min(sorted_points, axis=0)
    )

    max_distance = float("-inf")
    knee_point = sorted_points[0]
    knee_index = sorted_indices[0]

    # Loop over the points, compute the bend angle for each, and identify the point with the largest positive bend angle.
    for i in range(1, len(sorted_points) - 1):
        x = sorted_points[i]
        xR = sorted_points[i + 1]
        distance = distance_function(xR, x)
        if distance >= max_distance:
            max_distance = distance
            knee_point = x
            knee_index = sorted_indices[i]

    return knee_point, knee_index


if __name__ == "__main__":
    pareto_points = [[1, 5], [2, 4], [3, 2], [4, 1]]

    # Split the Pareto points into x and y coordinates
    x = [point[0] for point in pareto_points]
    y = [point[1] for point in pareto_points]

    # Plot the Pareto points
    plt.scatter(x, y, label="Pareto Points", color="b")

    # Add labels
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    # Show the legend
    plt.legend()

    # Display the grid
    plt.grid(True)
    plt.title("Pareto Frontier")
    plt.show()

    knee, index = find_manhattan_knee(pareto_points)
    print(f"Knee point: {knee}, Index: {index}")
