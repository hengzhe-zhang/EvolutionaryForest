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
    pf = front
    pf = (pf - np.min(pf, axis=0)) / (np.max(pf, axis=0) - np.min(pf, axis=0))
    p1 = pf[pf[:, 0].argmax()]
    p2 = pf[pf[:, 1].argmax()]
    # 自动选择拐点
    ans = max([i for i in range(len(pf))],
              key=lambda i: point_to_line_distance(p1, p2, pf[i]))
    return ans
