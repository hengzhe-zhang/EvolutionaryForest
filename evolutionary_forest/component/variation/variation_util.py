import math

from evolutionary_forest.component.configuration import CrossoverConfiguration


def get_number_of_invokes(gene_num, crossover_configuration: CrossoverConfiguration):
    if crossover_configuration.number_of_invokes > 0:
        invokes = crossover_configuration.number_of_invokes
        if invokes < 1:
            # based on ratio
            invokes = math.ceil(gene_num * invokes)
    else:
        invokes = gene_num
    return invokes
