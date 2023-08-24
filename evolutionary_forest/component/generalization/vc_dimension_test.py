from evolutionary_forest.component.generalization.vc_dimension import simple_argmin


def test_argmin():
    arr = [5, 2, 8, 1, 4]
    assert simple_argmin(arr) == 3

    arr = [9, 6, 3, 1, 7]
    assert simple_argmin(arr) == 3

    arr = [2, 4, 6, 8, 10]
    assert simple_argmin(arr) == 0

    arr = [1, 1, 1, 1, 1]
    assert simple_argmin(arr) == 0
