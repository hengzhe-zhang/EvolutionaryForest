from evolutionary_forest.component.crossover_mutation import intron_crossover, random_combination
from evolutionary_forest.multigene_gp import get_cross_point, get_intron_id


def crossover_based_on_intron(ind1, ind2, intron_parameters, pset):
    gene1, id1 = ind1.random_select(with_id=True)
    gene2, id2 = ind2.random_select(with_id=True)
    if intron_parameters.get('exon_tournament', False):
        # Warning: After mutation, there may exist some undefined parameters
        introns_a = [get_cross_point(gene1, min_tournament_size=2)]
        introns_b = [get_cross_point(gene2, min_tournament_size=2)]
        intron_crossover(gene1, gene2, introns_a, introns_b, cross_intron=False, avoid_fallback=True)
    elif intron_parameters.get("intron_crossover", False):
        # Using exons to replace introns
        introns_a, introns_b = get_intron_id(gene1), get_intron_id(gene2)
        intron_crossover(gene1, gene2, introns_a, introns_b, cross_intron=True)
    elif intron_parameters.get("random_combination", False):
        random_combination(gene1, gene2, pset)
    elif intron_parameters.get("exon_crossover", False):
        introns_a, introns_b = get_intron_id(gene1), get_intron_id(gene2)
        intron_crossover(gene1, gene2,
                         list(filter(lambda x: x not in introns_a, range(0, len(gene1)))),
                         list(filter(lambda x: x not in introns_b, range(0, len(gene2)))))
    else:
        raise Exception
