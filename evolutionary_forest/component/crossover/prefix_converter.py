class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []


def parse_expression_to_tree(expression):
    stack = []
    current_node = None

    for token in expression:
        if token in ["protected_div", "multiply"]:
            new_node = TreeNode(token)
            if current_node:
                stack.append(current_node)
            current_node = new_node
        else:
            if current_node:
                current_node.children.append(TreeNode(token))
                if len(current_node.children) == 2:
                    if stack:
                        parent = stack.pop()
                        parent.children.append(current_node)
                        current_node = parent
                    else:
                        break
    return current_node


def print_tree_prefix(node):
    result = []
    if node:
        result.append(node.value)
        for child in node.children:
            result.extend(print_tree_prefix(child))
    return result


# Example usage:
expression = ["protected_div", "ARG2", "multiply", "ARG3", "ARG9"]
tree = parse_expression_to_tree(expression)
print(print_tree_prefix(tree))
