from deap.gp import Primitive


class TreeNode:
    def __init__(self, val, children=None):
        self.val = val
        self.children = children if children else []


def convert_node_list_to_tree(tree: list, node_idx):
    if node_idx >= len(tree):
        return node_idx, None

    node = tree[node_idx]
    if isinstance(node, Primitive):
        tree_node = TreeNode(node)
        current_idx = node_idx + 1
        for _ in range(node.arity):
            if current_idx >= len(tree):
                break
            child_idx, child_node = convert_node_list_to_tree(tree, current_idx)
            tree_node.children.append(child_node)
            current_idx = child_idx
        return current_idx, tree_node
    else:
        # Assuming non-Primitive nodes are leaves or invalid, handle accordingly
        tree_node = TreeNode(node)
        return node_idx + 1, tree_node


def sort_son(root_tree: TreeNode):
    if root_tree is None:
        return None
    if root_tree.val.name in ["add", "multiply", "minimum", "maximum"]:
        root_tree.children = sorted(root_tree.children, key=lambda x: x.val.name)
    for child in root_tree.children:
        sort_son(child)
    return root_tree


def convert_tree_to_node_list(tree: TreeNode):
    if tree is None:
        return []
    result = [tree.val]
    for child in tree.children:
        result.extend(convert_tree_to_node_list(child))
    return result
