import copy
import random


def get_replacement_tree(ind, parent):
    # When having deletion or addition operators,
    # randomly selecting an unused tree for replacement is a reasonable solution.
    genes = set([str(gene) for gene in ind.gene])
    choice = list(filter(lambda x: str(x) not in genes, parent))
    if len(choice) == 0:
        # If no unused tree, just delete
        # Note: We cannot delete all unfeasible trees, because if all trees are deleted,
        # it will be problematic for evaluation.
        gene = None
    else:
        gene = copy.deepcopy(random.choice(choice))
    return gene


def remove_none_values(input_list):
    return [item for item in input_list if item is not None]
