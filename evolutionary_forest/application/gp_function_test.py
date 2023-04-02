import numpy as np  # noqa
from numpy import *
from baes_class import ExpensiveMOEABenchmark


def analytical_quotient(x1, x2):
    return x1 / np.sqrt(1 + (x2 ** 2))


class ExpensiveMOEABenchmark1(ExpensiveMOEABenchmark):
    def __init__(self):
        super().__init__()
        self.first_layer_funcs = [
            lambda X0: (maximum((add(X0, X0)), X0)),
            lambda X0: (add(X0, (negative(X0)))),
            lambda X0: (sin((add(X0, (add(X0, X0)))))),
            lambda X0: (cos((maximum(X0, X0)))),
            lambda X0: (negative((analytical_quotient(X0, X0)))),
        ]

        self.second_layer_funcs = [
            lambda VAR0, VAR1, VAR2, VAR3, VAR4: (multiply((cos(VAR2)), (cos(VAR1)))),
            lambda VAR0, VAR1, VAR2, VAR3, VAR4: (cos(VAR2)),
        ]


if __name__ == '__main__':
    bench = ExpensiveMOEABenchmark1()
    bench.evaluation()
    bench.plot()
