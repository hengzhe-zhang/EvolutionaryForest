import numpy as np


def bend_angle(x, xL, xR):
    """
    Compute the bend angle for a given point x with its two neighboring points xL and xR.
    """
    # valid in minimization Pareto front
    thetaL = np.arctan((xL[1] - x[1]) / (x[0] - xL[0]))
    thetaR = np.arctan((x[1] - xR[1]) / (xR[0] - x[0]))
    # print((xL[1] - x[1]), (x[0] - xL[0]), (x[1] - xR[1]), (xR[0] - x[0]))
    return thetaL - thetaR


def find_knee_based_on_bend_angle(pareto_points, local=False, four_neighbour=False):
    """
    Identify the knee point based on the bend angle.
    Returns both the knee point and its index.
    """
    # Sort pareto_points by the first objective in descending order while keeping track of original indices
    sorted_points_with_indices = sorted(enumerate(pareto_points), key=lambda x: x[1][0], reverse=False)
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
    # Loop over the points, compute the bend angle for each, and identify the point with the largest positive bend angle.
    for i in range(1, len(sorted_points) - 1):
        x = sorted_points[i]
        if local:
            xL = sorted_points[i - 1]
            xR = sorted_points[i + 1]
        theta = bend_angle(x, xL, xR)

        if four_neighbour:
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

            if angles_to_check:
                theta = max(angles_to_check)

        if theta > max_bend_angle:
            max_bend_angle = theta
            knee_point = x
            knee_index = sorted_indices[i]

    return knee_point, knee_index


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
