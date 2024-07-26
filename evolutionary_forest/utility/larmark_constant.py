from typing import List

import numpy as np
from deap.gp import PrimitiveTree, Terminal, Primitive, PrimitiveSet


def find_primitive_by_name(pset: PrimitiveSet, name: str) -> Primitive:
    for prim in pset.primitives[object]:
        if prim.name == name:
            return prim
    raise ValueError(f"Primitive '{name}' not found in the primitive set.")


def lamarck_constant(
    trees: List[PrimitiveTree], pset: PrimitiveSet, coefs: np.ndarray
) -> List[PrimitiveTree]:
    mul = find_primitive_by_name(pset, "Mul")

    for i, (tree, coef) in enumerate(zip(trees, coefs)):
        if isinstance(tree[0], Primitive) and tree[0].name == "Mul":
            if isinstance(tree[1], Terminal) and isinstance(
                tree[1].value, (float, int)
            ):
                tree[1] = Terminal(tree[1].value * coef, False, object)
                continue
        t = Terminal(coef, False, object)
        subtree = PrimitiveTree([mul, t] + list(tree))
        trees[i] = subtree
    return trees


if __name__ == "__main__":
    pset = PrimitiveSet("MAIN", 3)
    pset.addPrimitive(lambda x, y: x and y, 2, name="and")
    pset.addPrimitive(lambda x, y: x or y, 2, name="or")
    pset.addPrimitive(lambda x, y: x * y, 2, name="Mul")
    pset.addTerminal(1)

    trees = [PrimitiveTree.from_string("and(or(ARG0, ARG1), ARG2)", pset)]
    coefs = np.array([2])

    new_trees, labels = lamarck_constant(trees, pset, coefs)
    print(str(new_trees[0]))
    print(labels)
