import numpy as np

from evolutionary_forest.component.configuration import EQLHybridConfiguration
from evolutionary_forest.component.equation_learner.eql_util import symbolic_regression
from evolutionary_forest.component.equation_learner.sympy_parser import (
    convert_to_deap_gp,
)
from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utility.tree_size_counter import get_tree_size
from evolutionary_forest.utils import efficient_deepcopy

DEBUG = False


def eql_mutation(gp: MultipleGeneGP, pset, X, y, eql_config: EQLHybridConfiguration):
    # Create a deep copy of the GP individual.
    offspring = efficient_deepcopy(gp)

    # Retrieve the individual semantics (assumed shape: [n_samples, n_genes]).
    semantics = gp.semantics

    # Compute the overall prediction using the pipeline.
    result = gp.pipe.predict(semantics)  # shape: (n_samples,)

    # Retrieve the scaler and ridge regressor from the pipeline.
    scaler = gp.pipe["Scaler"]
    ridge = gp.pipe["Ridge"]

    # Choose a random gene index.
    n_genes = semantics.shape[1]
    random_idx = np.random.randint(0, n_genes)

    # Transform the semantics (assumed to be in shape (n_samples, n_genes)).
    scaled_semantics = scaler.transform(semantics)  # shape: (n_samples, n_genes)

    # Compute the contribution of the selected gene.
    # (ridge.coef_ is assumed to be an array with one coefficient per gene)
    gene_contrib = (
        ridge.coef_[random_idx] * scaled_semantics[:, random_idx]
    )  # shape: (n_samples,)

    # Remove the selected gene's contribution from the overall prediction.
    new_prediction = result - gene_contrib  # shape: (n_samples,)

    # Ensure that y is a flat array.
    y_flat = np.asarray(y).flatten()

    # Compute the residual that the selected gene should explain.
    residual = y_flat - new_prediction  # shape: (n_samples,)

    gp_tree = generate_tree_by_eql(X, residual, pset, eql_config)

    # Replace the gene at the selected index with the new GP tree.
    offspring.gene[random_idx] = gp_tree

    return offspring, random_idx


def generate_tree_by_eql(X, target, pset, eql_config):
    # Convert the resulting expression string to a DEAP GP tree.
    pset_dict = {
        v.name.lower(): v for v in pset.primitives[float] or pset.primitives[object]
    }
    pset_dict = {
        **pset_dict,
        "sin": pset_dict["rsin"],
        "cos": pset_dict["rcos"],
    }
    # Use symbolic regression on (X, residual) to evolve a new expression.
    # Reshape residual to (n_samples, 1) if needed.
    reg_weight = 5e-3
    while True:
        expr_str = symbolic_regression(
            X,
            target,
            reg_weight=reg_weight,
            verbose=False,
            n_layers=2,
        )

        gp_tree = convert_to_deap_gp(str(expr_str), pset, pset_dict)

        tree_size = get_tree_size(gp_tree)
        if tree_size <= 30:
            if DEBUG:
                print(f"Average size: {tree_size},{reg_weight}")
            break
        else:
            if DEBUG:
                print(f"Average size: {tree_size},{reg_weight}")
            reg_weight *= 2
    return gp_tree
