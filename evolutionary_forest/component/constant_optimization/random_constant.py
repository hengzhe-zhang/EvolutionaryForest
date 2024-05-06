import numpy as np

from evolutionary_forest.component.random_constant import random_int


def random_gaussian(std=1):
    return np.random.normal(loc=0, scale=std)


def constant_controller(type):
    if type is None:
        return None
    elif type == "Normal":
        return random_gaussian
    else:
        assert type == "Int"
        return random_int


if __name__ == "__main__":
    print(random_gaussian(1))
