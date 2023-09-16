from deap.gp import Primitive, Terminal
from deap.tools import selRandom

from evolutionary_forest.component.crossover_mutation import intron_crossover, random_combination


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


def get_intron_id(gene):
    introns_results = []
    for id, g in enumerate(gene):
        if isinstance(g, (IntronPrimitive, IntronTerminal)) and g.intron:
            introns_results.append(id)
    return introns_results


class IntronPrimitive(Primitive):
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq', 'corr', 'level', 'equal_subtree', 'hash_id')

    @property
    def intron(self):
        return self.corr < 0.01

    def __init__(self, name, args, ret):
        super().__init__(name, args, ret)
        self.corr = 0
        self.level = 0
        self.equal_subtree = -1
        self.hash_id = 0


class IntronTerminal(Terminal):
    __slots__ = ('name', 'value', 'ret', 'conv_fct', 'corr', 'level', 'hash_id')

    @property
    def intron(self):
        return self.corr < 0.01

    def __init__(self, terminal, symbolic, ret):
        super().__init__(terminal, symbolic, ret)
        self.corr = 0
        self.level = 0
        self.hash_id = 0


def get_cross_point(gene, inverse=False, min_tournament_size=1):
    # least min_tournament_size individuals
    tournsize = min(max(min_tournament_size, round(0.1 * len(gene))), len(gene))
    aspirants = selRandom(list([(k, getattr(g, 'corr', 0)) for k, g in enumerate(gene)]), tournsize)
    if inverse:
        point = min(aspirants, key=lambda x: x[1])
    else:
        point = max(aspirants, key=lambda x: x[1])
    return point[0]
