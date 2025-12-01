"""
Extended feature importance calculation using subtree semantics.

This module provides deep exploration of feature importance by:
1. Extracting all subtrees from GP trees
2. Evaluating their semantics
3. Merging features based on semantic hash (not structural hash)
4. Counting frequency based on semantic patterns
"""

from collections import defaultdict
from functools import lru_cache
from typing import TYPE_CHECKING
import numpy as np
import joblib
from deap.gp import PrimitiveTree
from sympy import latex, parse_expr

from evolutionary_forest.component.primitive_functions import individual_to_tuple
from evolutionary_forest.utility.model_analysis_util import model_simplification

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def _get_semantic_hash(
    semantic_vector,
    method="exact",
    lsh_bits=64,
    lsh_seed=0,
):
    """
    Compute a semantic hash from a semantic vector.
    Uses the same normalization as add_hash_value in evaluation.py.

    Parameters:
    -----------
    semantic_vector : np.ndarray
        The semantic output vector of a subtree

    Returns:
    --------
    int
        Hash value of the normalized semantic vector
    """
    if method == "lsh":
        return _get_lsh_semantic_hash(semantic_vector, n_bits=lsh_bits, seed=lsh_seed)

    if isinstance(semantic_vector, np.ndarray):
        if semantic_vector.var() != 0:
            normalized = (
                semantic_vector - semantic_vector.mean()
            ) / semantic_vector.var()
        else:
            normalized = semantic_vector - semantic_vector.mean()
        return joblib.hash(normalized)
    else:
        return hash(str(semantic_vector))


@lru_cache(maxsize=128)
def _get_lsh_random_matrix(length, n_bits, seed):
    """
    Initialize a random projection matrix for LSH hashing (cached).
    """
    rng = np.random.RandomState(seed + length + n_bits)
    return rng.randn(length, n_bits)


def _get_lsh_semantic_hash(semantic_vector, n_bits=64, seed=0):
    """
    Compute a locality-sensitive hash using random binary projections (SimHash).
    """
    if isinstance(semantic_vector, np.ndarray):
        normalized = semantic_vector - semantic_vector.mean()
        norm = np.linalg.norm(normalized)
        if norm > 0:
            normalized = normalized / norm
        else:
            normalized = np.zeros_like(normalized)
    else:
        normalized = np.asarray(semantic_vector)

    normalized = normalized.ravel()
    if normalized.size == 0:
        return 0

    random_matrix = _get_lsh_random_matrix(normalized.size, n_bits, seed)
    projections = normalized @ random_matrix
    signs = projections >= 0

    hash_value = 0
    for idx, flag in enumerate(signs):
        if flag:
            hash_value |= 1 << idx
    return hash_value


def _single_tree_evaluation_with_node_outputs(expr, pset, data, prefix="ARG"):
    """
    Evaluate a tree and return all node outputs (semantics of all subtrees).

    This function directly reuses the evaluation logic from single_tree_evaluation
    in evaluation.py by following the same pattern, but captures all intermediate
    node outputs instead of just the final result.

    Parameters:
    -----------
    expr : PrimitiveTree
        The GP tree to evaluate
    pset : PrimitiveSet
        The primitive set for evaluation
    data : np.ndarray
        Training data
    prefix : str
        Prefix for terminal names (default: "ARG")

    Returns:
    --------
    dict
        Dictionary mapping node_id to semantic output vector
    """
    from deap.gp import Primitive, Terminal

    node_outputs = {}  # Store semantic output for each node
    stack = []

    # Directly reuse the same evaluation pattern as single_tree_evaluation
    # (lines 845-988 in evaluation.py)
    for node_id, node in enumerate(expr):
        stack.append((node, [], node_id))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args, id = stack.pop()
            result = None

            if isinstance(prim, Primitive):
                try:
                    result = pset.context[prim.name](*args)
                except OverflowError:
                    result = args[0] if args else None
            elif isinstance(prim, Terminal):
                if prefix in prim.name:
                    if isinstance(data, np.ndarray) or hasattr(data, "__getitem__"):
                        result = data[:, int(prim.name.replace(prefix, ""))]
                    elif isinstance(data, (dict, type)):
                        result = data[prim.name]
                    else:
                        raise ValueError("Unsupported data type!")
                else:
                    if isinstance(prim.value, str):
                        result = float(prim.value)
                    else:
                        result = prim.value
            else:
                raise Exception(f"Unknown node type: {type(prim)}")

            # Store the semantic output for this node
            # (single_tree_evaluation computes this but only returns the final result)
            if result is not None:
                if isinstance(result, np.ndarray):
                    if len(result.shape) == 0:
                        result = np.array([result])
                    node_outputs[id] = result
                else:
                    # Convert scalar to array
                    node_outputs[id] = np.array([result])

            if len(stack) == 0:
                break
            stack[-1][1].append(result)

    return node_outputs


def _extract_subtree_semantics(gene, pset, X, tree_length_limit=None):
    """
    Extract all subtrees and their semantics from a GP tree in one pass.

    This function reuses the evaluation logic from evaluation.py to efficiently
    extract all subtree semantics by evaluating the tree once and capturing all
    intermediate node outputs.

    Parameters:
    -----------
    gene : PrimitiveTree
        The GP tree to extract subtrees from
    pset : PrimitiveSet
        The primitive set for evaluation
    X : np.ndarray
        Training data to evaluate subtrees on
    tree_length_limit : int, optional
        Maximum tree length to consider (None = no limit)

    Returns:
    --------
    list of tuples
        Each tuple contains (subtree_slice, subtree_tree, semantic_vector, node_id)
    """
    subtrees = []

    # Reuse evaluation logic to get all node outputs in one pass
    node_outputs = _single_tree_evaluation_with_node_outputs(gene, pset, X)

    # Now extract subtrees using the captured node outputs
    for node_id in range(len(gene)):
        # Get the subtree starting at this node
        subtree_slice = gene.searchSubtree(node_id)
        subtree_tree = PrimitiveTree(gene[subtree_slice])

        # Skip if tree length limit is set and exceeded
        if tree_length_limit is not None and len(subtree_tree) > tree_length_limit:
            continue

        # Get the semantic output for this node (which represents the subtree)
        if node_id in node_outputs:
            semantic_vector = node_outputs[node_id]
            subtrees.append((subtree_slice, subtree_tree, semantic_vector, node_id))

    return subtrees


def _gene_to_string(gene):
    """Convert a PrimitiveTree to string representation."""
    string = ""
    stack = []
    for node in gene:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            string = f"{prim.name}({','.join(args)})" if args else prim.name
            if len(stack) == 0:
                break
            stack[-1][1].append(string)
    return string if string else str(gene)


def _code_generation(regressor, tree):
    """Generate Python code from a tree."""
    code = str(tree)
    args = ",".join(arg for arg in regressor.pset.arguments)
    code = f"lambda {args}: {code}"
    return code


def get_feature_importance_semantic(
    regressor_model: "EvolutionaryForestRegressor",
    latex_version=True,
    fitness_weighted=False,
    mean_fitness=False,
    ensemble_weighted=True,
    simple_version=None,
    tree_length_limit=None,
    weight_by_length=False,
    hash_method="exact",
    lsh_bits=64,
    lsh_seed=0,
    include_subtrees=True,
    **params,
):
    """
    Extended feature importance calculation using subtree semantics.

    This function extracts all subtrees from each gene, evaluates their semantics,
    and merges features based on semantic hash (not structural hash). This allows
    semantically equivalent but structurally different expressions to be merged.

    Parameters:
    -----------
    regressor_model : EvolutionaryForestRegressor
        The trained evolutionary forest model
    latex_version : bool
        Return simplified LaTeX symbols for printing (default: True)
    fitness_weighted : bool
        Weight importance by individual fitness values (default: False)
    mean_fitness : bool
        Use mean importance instead of sum (default: False)
    ensemble_weighted : bool
        Weight by ensemble weights if available (default: True)
    simple_version : bool, optional
        Alias for latex_version
    tree_length_limit : int, optional
        Maximum tree length to consider (None = no limit)
    weight_by_length : bool
        Weight importance by 1/length to favor independent features over building blocks (default: False)
    hash_method : str
        Hashing strategy for semantics ("exact" or "lsh")
    lsh_bits : int
        Number of bits for LSH hashing when hash_method="lsh"
    lsh_seed : int
        Random seed for LSH hashing
    include_subtrees : bool
        If True, include all subtrees; if False, only consider full trees using existing hash_result
    **params
        Additional parameters

    Returns:
    --------
    dict
        Dictionary mapping feature expressions to normalized importance scores
    """
    if simple_version is not None:
        latex_version = simple_version

    if mean_fitness:
        all_genes_map = defaultdict(list)
    else:
        all_genes_map = defaultdict(float)

    # Dictionary to store the representative tree for each semantic hash
    hash_dict = {}
    # Dictionary to store tree lengths for each semantic hash
    hash_lengths = {}

    # Processing function for feature names
    if latex_version:
        processing_code = (
            lambda g: rf"{latex(parse_expr(model_simplification(_gene_to_string(g))), mode='inline')}"
        )
    else:
        processing_code = lambda g: f"{_code_generation(regressor_model, g)}"

    # Get training data for subtree evaluation
    X = regressor_model.X

    # Iterate through all individuals in hall of fame
    for x in regressor_model.hof:
        # Extract subtrees based on configuration
        for gene_idx, gene in enumerate(x.gene):
            # Get coefficient for this gene
            if gene_idx < len(x.coef):
                base_coef = np.abs(x.coef[gene_idx])
            else:
                base_coef = 1.0

            # Calculate importance value with optional weightings
            importance_value = base_coef
            if fitness_weighted:
                importance_value = importance_value * x.fitness.wvalues[0]

            if ensemble_weighted and hasattr(regressor_model.hof, "ensemble_weight"):
                importance_value = (
                    importance_value
                    * regressor_model.hof.ensemble_weight[individual_to_tuple(x)]
                )

            if include_subtrees:
                subtree_structures = _extract_subtree_semantics(
                    gene, regressor_model.pset, X, tree_length_limit
                )
            else:
                # Only consider the full tree, using existing hash_result
                if gene_idx >= len(x.hash_result):
                    continue
                if tree_length_limit is not None and len(gene) > tree_length_limit:
                    continue
                subtree_structures = [
                    (
                        gene.searchSubtree(0),
                        gene,
                        None,
                        0,
                    )
                ]

            for (
                subtree_slice,
                subtree_tree,
                semantic_vector,
                node_id,
            ) in subtree_structures:
                # Determine hash value source
                if include_subtrees:
                    semantic_hash = _get_semantic_hash(
                        semantic_vector,
                        method=hash_method,
                        lsh_bits=lsh_bits,
                        lsh_seed=lsh_seed,
                    )
                else:
                    semantic_hash = x.hash_result[gene_idx]

                # Calculate length-based weight if requested
                length_weight = 1.0
                if weight_by_length:
                    length_weight = 1.0 / len(subtree_tree)

                # Weighted importance for this subtree
                subtree_importance = importance_value * length_weight

                # Store representative tree (prefer shorter trees for same hash)
                if semantic_hash not in hash_dict or len(
                    subtree_tree
                ) < hash_lengths.get(semantic_hash, float("inf")):
                    hash_dict[semantic_hash] = subtree_tree
                    hash_lengths[semantic_hash] = len(subtree_tree)

                # Aggregate importance by semantic hash
                if mean_fitness:
                    all_genes_map[semantic_hash].append(subtree_importance)
                else:
                    all_genes_map[semantic_hash] += subtree_importance

    # Convert to feature names and compute means if needed
    all_genes_map = {
        processing_code(hash_dict[k]): all_genes_map[k] for k in all_genes_map.keys()
    }

    if mean_fitness:
        for k, v in all_genes_map.items():
            all_genes_map[k] = np.mean(v)

    # Sort by importance
    feature_importance_dict = {
        k: v for k, v in sorted(all_genes_map.items(), key=lambda item: -item[1])
    }

    # Normalize to sum to 1
    sum_value = np.sum([v for k, v in feature_importance_dict.items()])
    if sum_value > 0:
        feature_importance_dict = {
            k: v / sum_value for k, v in feature_importance_dict.items()
        }

    return feature_importance_dict
