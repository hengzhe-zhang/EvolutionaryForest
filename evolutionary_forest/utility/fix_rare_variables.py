import platform
from collections import Counter
from deap import gp


def get_variable_frequency(population, pset):
    """Calculate frequency of each variable across population (number of individuals using it)."""
    var_counter = Counter()
    for ind in population:
        used_vars = set()
        for tree in ind.gene:
            for node in tree:
                if isinstance(node, gp.Terminal):
                    if hasattr(node, "name") and node.name in pset.arguments:
                        used_vars.add(node.name)
        for var_name in used_vars:
            var_counter[var_name] += 1
    return var_counter


def is_rare_variable(var_name, var_frequency, pop_size, rare_threshold=0.1):
    """Check if variable is rare based on frequency threshold."""
    if var_name not in var_frequency:
        return True
    frequency_ratio = var_frequency[var_name] / pop_size
    return frequency_ratio < rare_threshold


def get_zero_terminal(pset):
    """Get or create a zero terminal."""
    for term in pset.terminals.get(float, []) + pset.terminals.get(object, []):
        if isinstance(term, gp.Terminal) and hasattr(term, "value"):
            try:
                if float(term.value) == 0.0:
                    return term
            except (ValueError, TypeError):
                pass
    return gp.Terminal(0.0, False, float)


def fix_rare_variables_in_tree(tree, var_frequency, pop_size, pset, rare_threshold=0.1):
    """Replace rare variable terminals with zero in a tree."""
    zero_terminal = get_zero_terminal(pset)
    modified = False
    for i, node in enumerate(tree):
        if isinstance(node, gp.Terminal):
            if hasattr(node, "name") and node.name in pset.arguments:
                if is_rare_variable(node.name, var_frequency, pop_size, rare_threshold):
                    tree[i] = zero_terminal
                    modified = True
    return modified


def fix_rare_variables_in_hof(hof, population, pset, rare_threshold=0.5):
    """Fix rare variables in Hall of Fame individuals."""
    if not hof or not population:
        return

    var_frequency = get_variable_frequency(population, pset)
    pop_size = len(population)

    if platform.system() == "Windows":
        print(f"Variable frequency counter: {dict(var_frequency)}")

    for ind in hof:
        for tree in ind.gene:
            fix_rare_variables_in_tree(
                tree, var_frequency, pop_size, pset, rare_threshold
            )
