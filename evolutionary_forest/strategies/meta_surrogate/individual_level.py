from evolutionary_forest.strategies.meta_surrogate.tree_level import *


def avg_tree_size(individual):
    return aggregate_trees(individual, tree_size, aggregation='average')


def max_tree_depth(individual):
    return aggregate_trees(individual, tree_depth, aggregation='max')


def avg_function_to_terminal_ratio(individual):
    return aggregate_trees(individual, function_to_terminal_ratio, aggregation='average')


def total_unique_functions(individual):
    return aggregate_trees(individual, unique_functions, aggregation='sum')


def total_unique_terminals(individual):
    return aggregate_trees(individual, unique_terminals, aggregation='sum')


def overall_function_usage(individual):
    """Aggregate function usage over all trees in an individual."""
    all_functions = [function for tree in individual for function in function_usage(tree).keys()]
    return dict(Counter(all_functions))


def overall_terminal_usage(individual):
    """Aggregate terminal usage over all trees in an individual."""
    all_terminals = [terminal for tree in individual for terminal in terminal_usage(tree).keys()]
    return dict(Counter(all_terminals))


def extract_global_vocab(individuals):
    """Extract a global vocabulary of all functions and terminals across all individuals."""
    all_functions = set()
    all_terminals = set()

    for ind in individuals:
        all_functions.update([func for tree in ind for func in function_usage(tree).keys()])
        all_terminals.update([term for tree in ind for term in terminal_usage(tree).keys()])

    return list(all_functions), list(all_terminals)


def vectorize_usage(individual, all_functions, all_terminals):
    """Convert function and terminal usage into a fixed-length feature vector."""
    func_usage = overall_function_usage(individual)
    term_usage = overall_terminal_usage(individual)

    # Only include counts for functions and terminals that existed in the training set
    func_vector = [func_usage.get(func, 0) for func in all_functions if func in func_usage]
    term_vector = [term_usage.get(term, 0) for term in all_terminals if term in term_usage]

    return func_vector + term_vector


def avg_function_diversity_index(individual):
    return aggregate_trees(individual, function_diversity_index, aggregation='average')


def avg_terminal_diversity_index(individual):
    return aggregate_trees(individual, terminal_diversity_index, aggregation='average')


def aggregate_trees(individual, metric_func, aggregation='average'):
    """
    Aggregate a metric over all trees in an individual.

    Args:
    - individual (list): A list of trees representing a GP individual.
    - metric_func (function): The metric function to apply to each tree.
    - aggregation (str): The type of aggregation ('average', 'sum', 'max', 'min').

    Returns:
    - Aggregated result.
    """
    results = [metric_func(tree) for tree in individual]

    if aggregation == 'average':
        return sum(results) / len(results)
    elif aggregation == 'sum':
        return sum(results)
    elif aggregation == 'max':
        return max(results)
    elif aggregation == 'min':
        return min(results)
    else:
        raise ValueError("Unknown aggregation type")
