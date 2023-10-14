from typing import List

import numpy as np
from deap import gp
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
    def __init__(self, max_depth=None, criterion='gini', splitter='best', random_state=None):
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

        self.column_transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), list(range(X.shape[1])))])
        X_encoded = self.column_transformer.fit_transform(X)

        self.tree = DecisionTreeClassifier(max_depth=self.max_depth, criterion=self.criterion,
                                           splitter=self.splitter, random_state=self.random_state)
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


def node_depths(tree, current_index=0, current_depth=1) -> (List[int], int):
    """
    Computes the depth of each node in a DEAP GP tree.

    Parameters:
    - tree: The DEAP GP tree.
    - current_index: The current node's index. Default is 0 for the tree root.
    - current_depth: The depth of the current node. Default is 1 for the tree root.

    Returns:
    - A list of depths for each node in the tree.
    """
    if current_index >= len(tree):
        return []

    depths = [current_depth]
    node = tree[current_index]

    # If the current node is a function (i.e., not a leaf), recurse into its children.
    if isinstance(node, gp.Primitive):
        offset = 1
        for _ in range(node.arity):
            child_depths, child_length = node_depths(tree, current_index + offset, current_depth + 1)
            depths.extend(child_depths)
            offset += child_length
        return depths, offset
    else:
        return depths, 1


# Example usage:
if __name__ == '__main__':
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(max, 2)
    pset.addTerminal(1)

    tree = gp.genFull(pset, min_=1, max_=3)
    print([str(n.name) for n in tree])
    depths, _ = node_depths(tree)
    print(depths)
    assert len(depths) == len(tree)
