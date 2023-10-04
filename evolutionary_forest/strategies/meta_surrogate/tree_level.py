import math
from collections import Counter

from deap import gp


def tree_size(tree):
    """Return the size of a tree."""
    return len(tree)


def tree_depth(tree):
    """Return the depth of a tree."""
    return tree.height


def function_to_terminal_ratio(tree):
    """Return the function-to-terminal ratio for a tree."""
    functions = [node for node in tree if isinstance(node, gp.Primitive)]
    terminals = [node for node in tree if isinstance(node, gp.Terminal)]
    return len(functions) / len(terminals)


def unique_functions(tree):
    """Return the count of unique functions in a tree."""
    functions = [node.name for node in tree if isinstance(node, gp.Primitive)]
    return len(set(functions))


def unique_terminals(tree):
    """Return the count of unique terminals in a tree."""
    terminals = [node.name for node in tree if isinstance(node, gp.Terminal)]
    return len(set(terminals))


def function_usage(tree):
    """Return a dictionary with the count of each function's usage."""
    functions = [node.name for node in tree if isinstance(node, gp.Primitive)]
    return dict(Counter(functions))


def terminal_usage(tree):
    """Return a dictionary with the count of each terminal's usage."""
    terminals = [node.name for node in tree if isinstance(node, gp.Terminal)]
    return dict(Counter(terminals))


def shannon_diversity_index(elements):
    """Return Shannon's Diversity Index for a list of elements."""
    counter = Counter(elements)
    probabilities = [count / len(elements) for count in counter.values()]
    return -sum(p * math.log2(p) for p in probabilities)


def function_diversity_index(tree):
    """Return Shannon's Diversity Index for functions in a tree."""
    functions = [node.name for node in tree if isinstance(node, gp.Primitive)]
    return shannon_diversity_index(functions)


def terminal_diversity_index(tree):
    """Return Shannon's Diversity Index for terminals in a tree."""
    terminals = [node.name for node in tree if isinstance(node, gp.Terminal)]
    return shannon_diversity_index(terminals)
