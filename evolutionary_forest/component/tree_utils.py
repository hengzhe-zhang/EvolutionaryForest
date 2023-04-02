import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


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


from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

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
