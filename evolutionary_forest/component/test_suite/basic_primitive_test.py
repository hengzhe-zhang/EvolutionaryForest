import numpy as np

from evolutionary_forest.component.primitive_functions import protected_division


def protected_division_test():
    # Test cases
    # Test case 1: Basic division with no near-zero values
    result = protected_division(10, 2)
    assert np.isclose(result, 5.0), f"Test case 1 failed: {result}"

    # Test case 2: Division by a very small positive value
    result = protected_division(10, 1e-12)
    assert np.isclose(result, 10 / 1e-10), f"Test case 2 failed: {result}"

    # Test case 3: Division by a very small negative value
    result = protected_division(10, -1e-12)
    assert np.isclose(result, 10 / -1e-10), f"Test case 3 failed: {result}"

    # Test case 4: Division by zero (should be protected by threshold)
    result = protected_division(10, 0)
    assert np.isclose(result, 10 / 1e-10), f"Test case 4 failed: {result}"

    print("All test cases passed!")


if __name__ == "__main__":
    # Run the tests
    protected_division_test()
