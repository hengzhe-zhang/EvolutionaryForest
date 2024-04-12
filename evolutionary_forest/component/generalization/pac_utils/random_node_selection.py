import random

import numpy as np
from deap import gp


# Redefine the random selection function to adapt to DEAP structures
def select_one_node_per_path(tree):
    paths = []

    def get_paths(node_idx, current_path):
        node = tree[node_idx]
        current_path.append(node_idx)
        if isinstance(node, gp.Primitive):
            if node.arity == 0:
                paths.append(list(current_path))
                return 1  # Size of leaf
            else:
                subtree_size = 1  # Starts with 1 to count the current node itself
                step = 0
                for arg in range(node.arity):  # Loop from 0 to arity-1
                    child_size = get_paths(
                        node_idx + 1 + step, list(current_path)
                    )  # Index of the child node
                    step += (
                        child_size  # Increment step by the size of the child's subtree
                    )
                    subtree_size += child_size  # Accumulate total subtree size
                return subtree_size
        else:
            paths.append(list(current_path))
            return 1  # Size of leaf

    get_paths(0, [])

    # print(paths)

    selected_nodes = set()
    forbidden_nodes = set()
    for path in paths:
        if any(node in selected_nodes for node in path):
            continue
        # Filter the path to remove any nodes that are forbidden
        filtered_path = [node for node in path if node not in forbidden_nodes]
        if filtered_path:
            chosen_node = random.choice(filtered_path)
            selected_nodes.add(chosen_node)
            for node in path:
                forbidden_nodes.add(node)

    return list(selected_nodes)


if __name__ == "__main__":
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(np.add, 2)
    pset.addTerminal(1)

    tree = gp.PrimitiveTree(gp.genGrow(pset, min_=2, max_=4))
    selected_nodes = select_one_node_per_path(tree)
    print(str(tree))
    print(selected_nodes)
