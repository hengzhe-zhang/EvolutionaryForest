import unittest

from euclidian_knee_selection import *


class TestPointToLineDistance(unittest.TestCase):

    def test_point_on_line(self):
        p1 = np.array([0, 0])
        p2 = np.array([1, 1])
        point = np.array([0.5, 0.5])
        self.assertAlmostEqual(point_to_line_distance(p1, p2, point), 0)

    def test_point_off_line(self):
        p1 = np.array([0, 0])
        p2 = np.array([1, 1])
        point = np.array([1, 0])
        self.assertAlmostEqual(point_to_line_distance(p1, p2, point), np.sqrt(0.5))

    def test_vertical_distance(self):
        p1 = np.array([0, 0])
        p2 = np.array([0, 2])
        point = np.array([1, 1])
        self.assertAlmostEqual(point_to_line_distance(p1, p2, point), 1)


class TestEuclidianKnee(unittest.TestCase):

    def test_basic_case(self):
        front = np.array([
            [1, 0],
            [0, 1],
            [0.3, 0.3],
            [0.2, 0.8],
            [0.8, 0.2]
        ])
        self.assertEqual(2, euclidian_knee(front))

    def test_single_point(self):
        front = np.array([
            [0.5, 0.5]
        ])
        # If only one point is given, it's considered the knee
        self.assertEqual(0, euclidian_knee(front))


if __name__ == '__main__':
    unittest.main()
