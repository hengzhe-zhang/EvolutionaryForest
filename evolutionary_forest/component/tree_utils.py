from typing import List

import numpy as np
from deap import gp
from deap.gp import PrimitiveTree
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.parent = None
        self.children = []


def construct_tree(inorder):
    val = inorder.pop(0)
    num_child = val.arity
    node = TreeNode(val)

    if num_child > 0:
        node.children = [construct_tree(inorder) for _ in range(num_child)]
        for child in node.children:
            child.parent = node
    return node


def get_parent_of_leaves(root: TreeNode):
    result = {}

    def traverse(node):
        if not node.children:
            # If the node is a leaf node, add its parent to the result dictionary
            result[node] = node.parent
        else:
            for child in node.children:
                traverse(child)

    traverse(root)
    return result


class StringDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, max_depth=None, criterion="gini", splitter="best", random_state=None
    ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.random_state = random_state
        self.tree: DecisionTreeClassifier = None
        self.label_encoder = None
        self.onehot_encoder = None
        self.column_transformer = None

    def fit(self, X, y, sample_weight=None):
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.column_transformer = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(), list(range(X.shape[1])))]
        )
        X_encoded = self.column_transformer.fit_transform(X)

        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            criterion=self.criterion,
            splitter=self.splitter,
            random_state=self.random_state,
        )
        self.tree.fit(X_encoded, y_encoded, sample_weight=sample_weight)

    def predict(self, X):
        X_encoded = self.column_transformer.transform(X)
        y_encoded = self.tree.predict(X_encoded)
        return self.label_encoder.inverse_transform(y_encoded)

    def predict_proba_sample(self, X):
        assert len(X) == 1
        y_encoded = self.predict_proba(X)
        if np.sum(y_encoded) == 0:
            label = np.random.choice(len(y_encoded[0]))
        else:
            label = np.random.choice(len(y_encoded[0]), p=y_encoded[0])
        return self.label_encoder.inverse_transform([label])[0]

    def predict_proba(self, X):
        X_encoded = self.column_transformer.transform(X)
        y_encoded = self.tree.predict_proba(X_encoded)
        return y_encoded


def node_depths(tree, current_index=0) -> (List[int], int):
    if current_index >= len(tree):
        return [], 0  # No depths and no length if outside the bounds of the tree

    node = tree[current_index]
    if isinstance(node, gp.Primitive):  # If it's a non-leaf node
        max_child_depth = 0  # Keep track of the maximum depth of the children
        offset = 1  # Starting offset for the first child
        all_child_depths = []
        for _ in range(node.arity):
            child_depths, child_length = node_depths(tree, current_index + offset)
            max_child_depth = max(
                max_child_depth, max(child_depths)
            )  # Update max depth based on children
            offset += child_length  # Move to the next child
            all_child_depths.extend(child_depths)
        depths = [
            max_child_depth + 1
        ] + all_child_depths  # Parent depth is 1 more than max child depth
        return depths, offset
    else:  # If it's a leaf node
        return [1], 1  # Leaf nodes have depth 1


# Example usage:
if __name__ == "__main__":
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(np.add, 2)
    pset.addTerminal(1)

    tree = gp.genGrow(pset, min_=2, max_=4)
    print([str(n.name) for n in tree])
    depths, _ = node_depths(PrimitiveTree(tree))
    print(depths)
    assert len(depths) == len(tree)
