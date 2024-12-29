from typing import List

from deap.gp import PrimitiveSet, Terminal

from evolutionary_forest.multigene_gp import MultipleGeneGP


def number_of_used_features(
    population: List[MultipleGeneGP], pset: PrimitiveSet
) -> int:
    possible_terminals = set([t.name for t in pset.terminals[object]])
    used_terminals = set()
    for h in population:
        for g in h.gene:
            for x in g:
                if isinstance(x, Terminal) and x.name in possible_terminals:
                    used_terminals.add(x.name)
    return len(used_terminals)
