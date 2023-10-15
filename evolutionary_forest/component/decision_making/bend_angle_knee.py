import numpy as np
from sklearn.cluster import KMeans


def bend_angle(x, xL, xR, weight=1):
    """
    Compute the bend angle for a given point x with its two neighboring points xL and xR.
    """
    # valid in minimization Pareto front
    thetaL = theta_left(x, xL)
    thetaR = theta_right(x, xR)
    # print((xL[1] - x[1]), (x[0] - xL[0]), (x[1] - xR[1]), (xR[0] - x[0]))
    return thetaL - weight * thetaR


def theta_right(x, xR):
    return np.arctan((x[1] - xR[1]) / (xR[0] - x[0]))


def theta_left(x, xL):
    return np.arctan((xL[1] - x[1]) / (x[0] - xL[0]))


def find_knee_based_on_bend_angle(pareto_points, local=False, four_neighbour=False,
                                  bend_weight=1, knee_selection=False):
    """
    Identify the knee point based on the bend angle.
    Returns both the knee point and its index.
    """
    # Sort pareto_points by the first objective in ascending order while keeping track of original indices
    # Model Error: Best->Worst
    # Model Complexity: Worst->Best
    sorted_points_with_indices = sorted(enumerate(pareto_points), key=lambda x: x[1][0])
    sorted_indices = [x[0] for x in sorted_points_with_indices]
    sorted_points = [x[1] for x in sorted_points_with_indices]

    if len(pareto_points) < 3:
        return sorted_points[0], sorted_indices[0]

    # min-max normalization
    sorted_points = np.array(sorted_points)
    sorted_points = ((sorted_points - np.min(sorted_points, axis=0)) /
                     (np.max(sorted_points, axis=0) - np.min(sorted_points, axis=0)))

    max_bend_angle = float('-inf')
    knee_point = sorted_points[0]
    knee_index = sorted_indices[0]

    xL = sorted_points[0]
    xR = sorted_points[-1]
    thetas = []
    # Loop over the points, compute the bend angle for each, and identify the point with the largest positive bend angle.
    for i in range(1, len(sorted_points) - 1):
        x = sorted_points[i]
        if local:
            xL = sorted_points[i - 1]
            xR = sorted_points[i + 1]
        theta = bend_angle(x, xL, xR, weight=bend_weight)

        if four_neighbour:
            theta = calculate_four_neighbour_max_angle(i, x, sorted_points)

        thetas.append(theta)
        if theta > max_bend_angle:
            max_bend_angle = theta
            knee_point = x
            knee_index = sorted_indices[i]

    # clustering-based knee detection
    if knee_selection is not False:
        thetas = np.array(thetas)
        thetas = thetas[~np.isnan(thetas)]
        n_clusters = knee_selection
        if len(thetas) >= n_clusters:
            kmeans = KMeans(n_clusters=int(n_clusters))
            kmeans.fit(thetas.reshape(-1, 1))
            cluster_centers = kmeans.cluster_centers_
            largest_x_cluster = np.argmax(cluster_centers)
            labels = kmeans.predict(thetas.reshape(-1, 1))
            first_knee = np.where(labels == largest_x_cluster)[0][-1]
            knee_index = sorted_indices[first_knee]
            knee_point = sorted_points[first_knee]

    return knee_point, knee_index


def calculate_four_neighbour_max_angle(i, x, sorted_points):
    """
       Calculate the maximum bend angle among four neighboring points for a given point in a sorted list.

       Args:
           i (int): Index of the point for which the maximum bend angle is calculated.
           x (np.ndarray): Current coordinate values.
           sorted_points (list of tuples): List of points.

       Returns:
           float: Maximum bend angle among the four neighboring points.
   """

    # Protection to avoid index errors
    angles_to_check = []
    if i - 1 >= 0 and i + 1 < len(sorted_points):
        angles_to_check.append(bend_angle(x, sorted_points[i - 1], sorted_points[i + 1]))
    if i - 1 >= 0 and i + 2 < len(sorted_points):
        angles_to_check.append(bend_angle(x, sorted_points[i - 1], sorted_points[i + 2]))
    if i - 2 >= 0 and i + 1 < len(sorted_points):
        angles_to_check.append(bend_angle(x, sorted_points[i - 2], sorted_points[i + 1]))
    if i - 2 >= 0 and i + 2 < len(sorted_points):
        angles_to_check.append(bend_angle(x, sorted_points[i - 2], sorted_points[i + 2]))
    theta = max(angles_to_check)
    return theta


def assert_decreasing_first_objective(pareto_points):
    for i in range(1, len(pareto_points)):
        assert pareto_points[i - 1][0] >= pareto_points[i][0], f"Point {i} is not in decreasing order!"


if __name__ == '__main__':
    pareto_points = [[1, 5], [2, 4], [3, 2], [4, 1]]
    knee, index = find_knee_based_on_bend_angle(pareto_points)
    print(f"Knee point: {knee}, Index: {index}")

    pareto_points = [[1, 5], [2, 4]]
    knee, index = find_knee_based_on_bend_angle(pareto_points)
    print(f"Knee point: {knee}, Index: {index}")
    assert index == 1
