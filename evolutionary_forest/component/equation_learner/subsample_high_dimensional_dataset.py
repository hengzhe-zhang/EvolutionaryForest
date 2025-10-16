import random

import numpy as np
from deap.gp import PrimitiveSet

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


def generate_forest_by_eql(X, target, pset, eql_config):
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

    non_constant_indices = _get_non_constant_indices(X)

    if len(non_constant_indices) == 0:
        return []

    threshold = 5
    if n_features <= threshold:
        # Low-dimensional: use all non-constant features
        gp_tree = _generate_tree_with_feature_subset(
            X, target, pset, eql_config, non_constant_indices, orig_names
        )
        trees.append(gp_tree)
    else:
        # High-dimensional: perform five rounds, each with 5 randomly sampled features.
        rounds = 5
        for _ in range(rounds):
            # Randomly select sample_size feature indices without replacement.
            sampled_indices = random.sample(non_constant_indices, threshold)
            gp_tree_mapped = _generate_tree_with_feature_subset(
                X, target, pset, eql_config, sampled_indices, orig_names
            )
            trees.append(gp_tree_mapped)

    return trees


def _get_non_constant_indices(X):
    # Identify constant features (features with zero variance)
    variances = np.var(X, axis=0)
    constant_features = np.where(variances == 0)[0]

    # Remove constant features from the dataset
    non_constant_indices = [i for i in range(X.shape[1]) if i not in constant_features]
    return non_constant_indices


def _generate_tree_with_feature_subset(
    X, target, pset, eql_config, feature_indices, original_feature_names
):
    # Build a feature map from DEAP's pset terminals (for object and float types).
    # This creates a dictionary mapping from terminal name (e.g., "ARG0") to the actual terminal object.
    feature_map = {
        v.name: v
        for v in (pset.terminals[object] + pset.terminals[float])
        if not v.name.startswith("rand")
    }

    no_float = False
    if len(pset.terminals[float]) == 0:
        del pset.terminals[float]
        no_float = True

    # Subset X based on type (DataFrame or numpy array).
    X_sub = X[:, feature_indices]

    # Run the original EQL procedure on the sampled features.
    gp_tree_sub = generate_tree_by_eql(X_sub, target, pset, eql_config)

    # Build a mapping from the GP tree's terminal names ("ARG0", "ARG1", ...)
    # to the corresponding DEAP pset terminal objects.
    mapping = {
        pset.arguments[i]: feature_map[original_feature_names[sampled_idx]]
        for i, sampled_idx in enumerate(feature_indices)
    }

    # Map the feature names in the gp_tree back to the DEAP pset terminal objects.
    gp_tree_mapped = map_features_in_tree(gp_tree_sub, mapping)

    if no_float:
        for node in gp_tree_mapped:
            if node.ret == float:
                node.ret = object
    return gp_tree_mapped


def generate_forest_by_bootstrap(X, target, pset, eql_config, n_rounds=10):
    """
    New bootstrap strategy: sample both samples and features (if high-dim).

    Parameters:
    - n_rounds: Number of bootstrap rounds (default: 5)
    """
    trees = []
    n_samples, n_features = X.shape

    # Get non-constant features
    non_constant_indices = _get_non_constant_indices(X)
    if len(non_constant_indices) == 0:
        return []

    original_feature_names = [f"ARG{i}" for i in range(n_features)]
    threshold = 5

    for _ in range(n_rounds):
        # Bootstrap sample indices (with replacement)
        sample_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[sample_indices]
        target_boot = target[sample_indices]

        # Determine feature indices to use
        if len(non_constant_indices) <= threshold:
            # Low-dimensional: use all non-constant features
            feature_indices = non_constant_indices
        else:
            # High-dimensional: randomly sample threshold features
            feature_indices = random.sample(non_constant_indices, threshold)

        # Generate tree with bootstrapped data and selected features
        tree = _generate_tree_with_feature_subset(
            X_boot,
            target_boot,
            pset,
            eql_config,
            feature_indices,
            original_feature_names,
        )
        trees.append(tree)

    return trees


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from evolutionary_forest.component.configuration import EQLHybridConfiguration
    from evolutionary_forest.utility.feature_selection import remove_constant_variables

    X, y = make_regression(n_features=5)
    X[:, 2] = np.ones(X.shape[0])  # Add a constant feature
    pset = PrimitiveSet("MAIN", X.shape[1])
    pset.addPrimitive(np.sin, 1, "sinpi")
    pset.addPrimitive(np.cos, 1, "cospi")
    pset.addPrimitive(np.add, 2, "add")
    pset.addPrimitive(np.subtract, 2, "sub")
    pset.addPrimitive(np.multiply, 2, "mul")
    pset.addPrimitive(np.divide, 2, "div")
    pset.addPrimitive(np.square, 2, "square")
    pset.addPrimitive(np.negative, 2, "neg")
    remove_constant_variables(pset, X)
    for tree in generate_forest_by_bootstrap(X, y, pset, EQLHybridConfiguration()):
        print(tree)
