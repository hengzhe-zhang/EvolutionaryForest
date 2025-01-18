from typing import Union, Callable

from deap.base import Toolbox
from deap.gp import PrimitiveSet

from evolutionary_forest.multigene_gp import MultiplePrimitiveSet


class TypedToolbox(Toolbox):
    root_crossover: bool
    pset: Union[MultiplePrimitiveSet, PrimitiveSet]
    # for mutation
    expr_mut: Callable
    # generate trees with flexible height
    tree_generation: Callable
    # for initialization
    expr: Callable
    # for initialisation
    individual: Callable
    # for evaluation
    evaluate: Callable
    # for compilation
    compile: Callable
    # for crossover
    mate: Callable
    # for mutation
    mutate: Callable
    # population
    population: Callable
    # clone
    clone: Callable


if __name__ == "__main__":
    box = TypedToolbox()
    box.expr = 1
    print(box.expr)
