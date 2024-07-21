import copy

from deap import gp


def copy_and_rename_tree(original_tree):
    # Copy the tree to avoid modifying the original
    tree_copy = copy.deepcopy(original_tree)

    # Step 1: Collect used features in order
    used_features = set()
    for node in original_tree:
        if isinstance(node, gp.Terminal) and node.name.startswith("ARG"):
            feature_id = int(node.name.replace("ARG", ""))
            if feature_id not in used_features:
                used_features.add(feature_id)
    used_features = list(sorted(list(used_features)))

    # Step 2: Create a mapping from original feature names to new feature names
    mapping_dict = {
        f"ARG{feature_id}": f"ARG{i}" for i, feature_id in enumerate(used_features)
    }
    mapping_dict_copy = copy.deepcopy(mapping_dict)

    # Step 3: Rename features in the copied tree based on the mapping
    for node in tree_copy:
        if isinstance(node, gp.Terminal) and node.name in mapping_dict_copy:
            # Only rename once
            del mapping_dict_copy[node.name]
            node.name = mapping_dict[node.name]
            node.value = mapping_dict[node.value]

    return tree_copy, used_features, mapping_dict
