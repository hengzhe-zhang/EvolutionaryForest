from typing import Callable

from deap.gp import mutInsert

from evolutionary_forest.component.crossover_mutation import intron_mutation
from evolutionary_forest.multigene_gp import get_cross_point, get_intron_id


def mutation_based_on_intron(individual, expr: Callable, intron_parameters, pset):
    gene, id = individual.random_select(with_id=True)
    if intron_parameters.get('exon_tournament', False):
        introns = [get_cross_point(gene, inverse=True, min_tournament_size=2)]
        intron_mutation(gene, expr, pset, introns)
    elif intron_parameters.get('mutation_worst', False):
        introns = [min(list([(k, getattr(g, 'corr', 0)) for k, g in enumerate(gene)]), key=lambda x: x[1])[0]]
        intron_mutation(gene, expr, pset, introns)
    elif intron_parameters.get('intron_mutation', False):
        introns = get_intron_id(gene)
        intron_mutation(gene, expr, pset, introns)
    elif intron_parameters.get('insert_mutation', False):
        mutInsert(gene, pset)
    elif intron_parameters.get('exon_mutation', False):
        # only mutate exons
        introns = get_intron_id(gene)
        list_of_exons = list(filter(lambda x: x not in introns, range(0, len(gene))))
        intron_mutation(gene, expr, pset, list_of_exons)
    else:
        raise Exception
    return id
