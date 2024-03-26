from deap.gp import Terminal

from evolutionary_forest.multigene_gp import MultipleGeneGP


def simple_simplification(ind: MultipleGeneGP):
    visited = set()
    all_genes = []
    for gene in ind.gene:
        # check constant tree
        constant = False
        for node in gene:
            if isinstance(node, Terminal) and isinstance(node.value, str):
                constant = False
                break
        if constant:
            continue

        # check constructed tree
        if str(gene) not in visited:
            visited.add(str(gene))
            all_genes.append(gene)
    if len(all_genes) != 0:
        ind.gene = all_genes
