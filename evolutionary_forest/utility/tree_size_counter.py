from deap.gp import PrimitiveTree, Terminal
from sklearn.base import TransformerMixin


def get_tree_size(gp: PrimitiveTree):
    return len(
        [
            node
            for node in gp
            if (
                # SAM-GP, smooth parameter is not counted
                not (
                    isinstance(node, Terminal)
                    and isinstance(node.value, TransformerMixin)
                )
            )
        ]
    )
