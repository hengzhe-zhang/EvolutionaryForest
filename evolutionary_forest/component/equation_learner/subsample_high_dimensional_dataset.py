import random

from evolutionary_forest.component.equation_learner.gp_util import generate_tree_by_eql


def map_features_in_tree(gp_tree, mapping):
    """
    Walks through the gp_tree and replaces terminal node names based on the given mapping.
    It is assumed that terminal nodes have a 'name' attribute or can be converted to a string that matches
    the keys in mapping.
    """
    for i, node in enumerate(gp_tree):
        if str(node.name) in mapping:
            # Otherwise, replace the node directly.
            gp_tree[i] = mapping[node.name]
    return gp_tree


def generate_trees_by_eql_ensemble(X, target, pset, eql_config):
    """
    Wrapper function for generate_tree_by_eql.

    - For datasets with <= 5 features: run EQL once on the full dataset.
    - For datasets with > 5 features: run EQL five times using a random sample of 5 features per round.

    After each run, the function maps the terminal names in the resulting GP tree (assumed to be named "ARG0", "ARG1", ...)
    back to the corresponding DEAP pset terminal objects (as required by DEAP).

    Returns a list of gp_tree.
    """
    trees = []

    # Determine the number of features.
    try:
        n_features = X.shape[1]
    except AttributeError:
        raise ValueError("X must be a 2D array-like structure with a shape attribute.")

    # For this implementation, we assume that the dataset's features are identified as "ARG0", "ARG1", etc.
    orig_names = [f"ARG{i}" for i in range(n_features)]

    # Build a feature map from DEAP's pset terminals (for object and float types).
    # This creates a dictionary mapping from terminal name (e.g., "ARG0") to the actual terminal object.
    feature_map = {v.name: v for v in (pset.terminals[object] + pset.terminals[float])}

    if n_features <= 5:
        # Low-dimensional: run on the full dataset.
        gp_tree = generate_tree_by_eql(X, target, pset, eql_config)
        trees.append(gp_tree)
    else:
        # High-dimensional: perform five rounds, each with 5 randomly sampled features.
        rounds = 5
        sample_size = 5
        for _ in range(rounds):
            # Randomly select sample_size feature indices without replacement.
            sampled_indices = random.sample(range(n_features), sample_size)

            # Subset X based on type (DataFrame or numpy array).
            X_sub = X[:, sampled_indices]

            # Run the original EQL procedure on the sampled features.
            gp_tree_sub = generate_tree_by_eql(X_sub, target, pset, eql_config)

            # Build a mapping from the GP tree's terminal names ("ARG0", "ARG1", ...)
            # to the corresponding DEAP pset terminal objects.
            mapping = {
                f"ARG{i}": feature_map[orig_names[sampled_idx]]
                for i, sampled_idx in enumerate(sampled_indices)
            }

            # Map the feature names in the gp_tree back to the DEAP pset terminal objects.
            gp_tree_mapped = map_features_in_tree(gp_tree_sub, mapping)
            trees.append(gp_tree_mapped)

    return trees
