from deap.tools import sortNondominated

from evolutionary_forest.component.configuration import EQLHybridConfiguration
from evolutionary_forest.component.equation_learner.gp_util import eql_mutation

DEBUG = True


def eql_hybrid_on_pareto_front(
    population, X, y, primitive_set, depth_limit, eql_config: EQLHybridConfiguration
):
    if eql_config.eql_hybrid_on_pareto_front == 0:
        return []
    pareto_fronts = sortNondominated(
        population, eql_config.eql_hybrid_on_pareto_front, first_front_only=True
    )[0]
    new_list = []
    # Apply mutation and respect depth limit
    avg_size = []
    for ind in pareto_fronts:
        eql_ind, random_idx = eql_mutation(ind, primitive_set, X, y, eql_config)
        avg_size.append(len(eql_ind.gene[random_idx]))
        if eql_ind.gene[random_idx].height > depth_limit:
            continue
        else:
            del eql_ind.fitness.values  # Reset fitness
            new_list.append(eql_ind)
    if DEBUG:
        print("Number of EQL hybrid on pareto front: ", len(new_list))
        print(
            f"Average size of EQL hybrid on pareto front: {sum(avg_size) / len(avg_size)},"
            f"{eql_config.reg_weight}"
        )
    return new_list


def eql_hybrid_on_best(best_individual, X, y, primitive_set, depth_limit, config):
    eql_ind, random_idx = eql_mutation(best_individual, primitive_set, X, y, config)
    if eql_ind.gene[random_idx].height <= depth_limit:
        del eql_ind.fitness.values  # Reset fitness
        return [eql_ind]
    else:
        return []
