import numpy as np
from scipy.stats import ranksums


def statistical_difference_between_populations(offspring, population):
    num_top_individuals = 30
    p_value = ranksums(
        [
            o.fitness.wvalues[0]
            for o in sorted(
                offspring, key=lambda x: x.fitness.wvalues[0], reverse=True
            )[:num_top_individuals]
        ],
        [
            o.fitness.wvalues[0]
            for o in sorted(
                population, key=lambda x: x.fitness.wvalues[0], reverse=True
            )[:num_top_individuals]
        ],
    )
    return p_value[1]


def check_number_of_unique_tree_semantics(offspring, num_of_trees):
    print(
        "Unique Hash",
        [
            len(
                np.unique(
                    [o.hash_result[i] for o in offspring if i < len(o.hash_result)]
                )
            )
            for i in range(num_of_trees)
        ],
    )
