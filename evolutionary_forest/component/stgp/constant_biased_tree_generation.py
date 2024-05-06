import operator
import sys
from typing import Callable
import random  # noqa
from evolutionary_forest.component.constant_optimization.random_constant import (
    random_gaussian,
)
from evolutionary_forest.component.stgp.shared_type import FeatureLayer


def genFull_STGP_constant_biased(pset, min_, max_, constant_prob, type_=None):
    def condition(height, depth):
        return depth >= height

    return generate_STGP_constant_biased(
        pset, min_, max_, condition, type_, constant_prob
    )


def genGrow_STGP_constant_biased(pset, min_, max_, constant_prob, type_=None):
    def condition(height, depth):
        return depth >= height or (
            depth >= min_ and random.random() < pset.terminalRatio
        )

    return generate_STGP_constant_biased(
        pset, min_, max_, condition, type_, constant_prob
    )


def genHalfAndHalf_STGP_constant_biased(pset, min_, max_, constant_ratio, type_=None):
    method = random.choice((genGrow_STGP_constant_biased, genFull_STGP_constant_biased))
    return method(pset, min_, max_, constant_ratio, type_)


def generate_STGP_constant_biased(
    pset, min_, max_, condition, type_=None, constant_prob=0.2
):
    if any(primitive == FeatureLayer for primitive in pset.primitives.keys()):
        # May need to increase one layer because FeatureLayer is not counted
        min_ += 1
        max_ += 1
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth) and type_ != FeatureLayer:
            try:
                # Decide whether to choose a constant or other terminal
                if random.random() < constant_prob and pset.terminals[type_]:
                    # Filter constants from terminals
                    constants = [
                        t
                        for t in pset.terminals[type_]
                        if isinstance(t, (int, float, Callable))
                    ]
                    if constants:
                        term = random.choice(constants)
                    else:
                        term = random.choice(pset.terminals[type_])
                else:
                    # Choose from non-constant terminals
                    non_constants = [
                        t
                        for t in pset.terminals[type_]
                        if not isinstance(t, (int, float))
                    ]
                    term = random.choice(
                        non_constants if non_constants else pset.terminals[type_]
                    )
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a terminal of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            if callable(term):
                term = term()
            expr.append(term)
        else:
            try:
                if len(pset.primitives[type_]) == 0:
                    term = random.choice(pset.terminals[type_])
                    if callable(term):
                        term = term()
                    expr.append(term)
                    continue
                else:
                    prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError(
                    "The gp.generate function tried to add "
                    "a primitive of type '%s', but there is "
                    "none available." % (type_,)
                ).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr


if __name__ == "__main__":
    import random
    from deap import gp, base, creator

    # Define a primitive set for symbolic regression
    pset = gp.PrimitiveSet("MAIN", arity=1)  # Assume one input variable
    pset.addPrimitive(operator.add, 2)  # Addition operator
    pset.addPrimitive(operator.sub, 2)  # Subtraction operator
    pset.addPrimitive(operator.mul, 2)  # Multiplication operator

    # Adding some terminals
    pset.addEphemeralConstant("rand101", random_gaussian)

    # Define a condition to control the depth
    def condition(height, depth):
        return depth == height

    # Define the type of individuals
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Example usage of the modified generate_STGP function
    def demo_generate_STGP():
        min_depth = 2
        max_depth = 5
        tree = generate_STGP_constant_biased(pset, min_depth, max_depth, condition)
        return tree

    # Generate and print the example tree
    example_tree = demo_generate_STGP()
    print(example_tree)
    print(str(gp.PrimitiveTree(example_tree)))
