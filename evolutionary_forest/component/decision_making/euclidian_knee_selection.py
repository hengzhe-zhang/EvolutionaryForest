import numpy as np


def point_to_line_distance(A, B, P):
    """
    Compute the shortest distance between point P and the line defined by points A and B.

    Args:
    -  A, B, P: 2D numpy arrays representing the points

    Returns:
    - Distance from point P to the line AB
    """

    # Ensure the input points are numpy arrays
    P = np.array(P)
    A = np.array(A)
    B = np.array(B)

    # Compute vectors
    AP = P - A
    AB = B - A

    # Compute the cross-product of AP and AB
    cross_product = np.linalg.norm(np.cross(AP, AB))

    # Compute the length of vector AB
    length_AB = np.linalg.norm(AB)

    # Compute the distance from point P to line AB
    distance = cross_product / length_AB

    return distance


def euclidian_knee(front):
    # Normalize the front
    pf = (front - np.min(front, axis=0)) / (
        np.max(front, axis=0) - np.min(front, axis=0)
    )

    # Enumerate points for later recovery of original order
    indexed_pf = list(enumerate(pf))

    # Sort the points by the first objective
    sorted_pf = sorted(indexed_pf, key=lambda x: x[1][0])

    # Extract sorted points and indices separately
    indices, sorted_points = zip(*sorted_pf)
    sorted_points = np.array(sorted_points)

    # Find points with the maximum values for each objective
    p1 = sorted_points[sorted_points[:, 0].argmax()]
    p2 = sorted_points[sorted_points[:, 1].argmax()]

    # Identify the knee point
    knee_index = max(
        [i for i in range(len(sorted_points))],
        key=lambda i: point_to_line_distance(p1, p2, sorted_points[i]),
    )

    # Return the index of the knee point in the original order
    return indices[knee_index]
