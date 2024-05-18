import random
from typing import Callable

from deap.algorithms import varAnd
from deap.gp import mutUniform

from evolutionary_forest.component.configuration import SelectionMode
from evolutionary_forest.component.primitive_functions import individual_to_tuple
from evolutionary_forest.multigene_gp import (
    mapElitesCrossover,
    semanticFeatureCrossover,
)
from evolutionary_forest.utils import efficient_deepcopy, is_float


def perform_semantic_macro_crossover(offspring, config, toolbox, y):
    parent = [offspring[0], offspring[1]]
    offspring = [efficient_deepcopy(offspring[0]), efficient_deepcopy(offspring[1])]

    if config.semantic_selection_mode == SelectionMode.AngleDriven:
        offspring = semanticFeatureCrossover(offspring[0], offspring[1], target=y)
    elif config.semantic_selection_mode == SelectionMode.MAPElites:
        offspring = mapElitesCrossover(
            offspring[0],
            offspring[1],
            target=y,
            map_elites_configuration=config.map_elites_configuration,
        )
    else:
        raise Exception("Invalid Selection Mode!")

    return offspring, parent


def handle_tpot_base_learner_mutation(offspring, base_learner, tpot_model):
    if base_learner == "Hybrid":
        base_models = [o.base_model for o in offspring]
        base_models = varAnd(base_models, tpot_model._toolbox, 0.5, 0.1)
        for o, b in zip(offspring, base_models):
            o.base_model = b
    return offspring


def check_redundancy_and_fix(
    offspring, bloat_control, toolbox, pset, new_tree_generation: Callable
):
    check_mutation = bloat_control.get("check_mutation", False)
    check_initialization = bloat_control.get("check_initialization", False)

    if check_mutation or check_initialization:
        for o in offspring:
            previous_set = set()
            for id, gene in enumerate(o.gene):
                if str(gene) in previous_set or is_float(str(gene)):
                    if check_mutation:
                        o.gene[id] = mutUniform(gene, toolbox.expr_mut, pset)[0]
                    if check_initialization:
                        o.gene[id] = new_tree_generation()
                previous_set.add(str(gene))

    return offspring


def norevisit_strategy_handler(
    offspring,
    toolbox,
    norevisit_strategy,
    evaluated_individuals,
    gene_addition_function,
):
    temp_evaluated_individuals = set()
    if norevisit_strategy == "Mutation":
        for o in offspring:
            while individual_to_tuple(o) in evaluated_individuals:
                o = toolbox.mutate(o)[0]
                temp_evaluated_individuals.add(individual_to_tuple(o))
    elif norevisit_strategy == "MutationOrAdditionOrDeletion":
        for o in offspring:
            while individual_to_tuple(o) in evaluated_individuals:
                r = random.random()
                if r < 1 / 3:
                    o = toolbox.mutate(o)[0]
                elif r < 2 / 3:
                    gene_addition_function(o)
                else:
                    o.gene_deletion()
