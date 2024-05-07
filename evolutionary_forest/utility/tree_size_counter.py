from deap.gp import PrimitiveTree, Terminal
from sklearn.base import TransformerMixin


def get_tree_size(gp: PrimitiveTree):
    return len(
        [
            node
            for node in gp
            if not isinstance(node, Terminal)
            or not isinstance(node.value, TransformerMixin)
        ]
    )
