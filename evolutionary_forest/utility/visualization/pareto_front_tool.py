import numpy as np
import matplotlib.pyplot as plt


def plot_pareto_front(population):
    """
    Plots points and highlights the non-dominated points (Pareto front).

    Args:
    - parent_a (list of dict): List of individuals, where each individual is a dictionary
      with 'fitness' containing a tuple of objective values.
    """
    # Extract the points
    points = np.array(
        [(ind.fitness.wvalues[0], ind.fitness.wvalues[1]) for ind in population]
    )

    # Function to find non-dominated points
    def pareto_front(points):
        is_dominated = np.zeros(len(points), dtype=bool)
        for i in range(len(points)):
            for j in range(len(points)):
                if i != j:
                    if (
                        points[j, 0] <= points[i, 0] and points[j, 1] < points[i, 1]
                    ) or (points[j, 0] < points[i, 0] and points[j, 1] <= points[i, 1]):
                        is_dominated[i] = True
                        break
        return ~is_dominated

    # Get non-dominated points (Pareto front)
    pareto_indices = pareto_front(points)
    pareto_points = points[pareto_indices]

    # Plotting
    plt.figure(figsize=(10 * 0.6, 6 * 0.6))
    plt.scatter(points[:, 0], points[:, 1], color="blue", label="Points")
    plt.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        color="red",
        label="Pareto Front",
        edgecolor="black",
    )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title("Points and Pareto Front")
    plt.legend()
    plt.grid(True)
    plt.show()
