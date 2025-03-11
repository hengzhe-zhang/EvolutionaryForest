from deap import gp
from deap.gp import PrimitiveSet


def mark_weights_only_terminal(pop, pset: PrimitiveSet):
    for ind in pop:
        for tree in ind.gene:
            tree: gp.PrimitiveTree
            weights = []
            for idx in range(len(tree)):
                sub_idx = tree.searchSubtree(idx)
                sum_weight = 0
                weight = 0
                for sub in range(sub_idx.start, sub_idx.stop):
                    node = tree[sub]
                    if isinstance(node, gp.Terminal):
                        sum_weight += 1

                    if isinstance(node, gp.Terminal) and node in pset.terminals[object]:
                        weight += 1
                    elif isinstance(node, gp.Terminal) and isinstance(
                        node.value, (int, float)
                    ):
                        # constant terminal
                        weight += 1

                # Avoid division by zero
                subtree_weight = weight / sum_weight if sum_weight > 0 else 0
                weights.append(subtree_weight)

            # Attach computed weights to the tree
            tree.subtree_weights = weights
