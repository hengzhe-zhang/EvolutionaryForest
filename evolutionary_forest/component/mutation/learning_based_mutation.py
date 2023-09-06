import operator

from deap.gp import PrimitiveTree, Primitive, Terminal
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from deap import gp, base, creator, tools
import random
from evolutionary_forest.multigene_gp import MultipleGeneGP


def parse_tree_to_tuples(root: PrimitiveTree):
    stack = []
    tuples = []

    for node in root:  # 假设iter_tree是一个用于遍历树的迭代器
        children = []
        stack.append((node, children))

        while len(stack[-1][1]) == stack[-1][0].arity:
            parent, child_list = stack.pop()
            tuple_entry = (parent, *child_list)
            tuples.append(tuple_entry)

            if len(stack) == 0:
                break

            stack[-1][1].append(parent)

    return tuples


def random_non_leaf_node(individual: PrimitiveTree):
    non_leaf_indices = [index for index, node in enumerate(individual) if isinstance(node, Primitive)]
    if len(non_leaf_indices) == 0:
        return None
    random_index = random.choice(non_leaf_indices)
    return random_index


def get_name(parent):
    if isinstance(parent, Primitive):
        name = parent.name
    elif isinstance(parent, Terminal):
        name = parent.value
    else:
        raise Exception
    return name


# 1. Many models can be used
class BuildingBlockLearning():
    def __init__(self):
        self.decision_tree = Pipeline([
            ('OneHot', OneHotEncoder()),
            ('DT', DecisionTreeClassifier()),
        ])
        self.label_encoder = LabelEncoder()

    def fit(self, individual: MultipleGeneGP):
        for g in individual.gene:
            self.fit_tree(g)

    def fit_tree(self, g):
        tuples = parse_tree_to_tuples(g)
        training_data = []
        training_label = []
        for tp in tuples:
            for i in range(1, len(tp)):
                training_data.append([get_name(t) for t in tp[:i - 1]] + ['?'] +
                                     [get_name(t) for t in tp[i + 1:]])
        training_label = self.label_encoder.fit(training_label)
        self.decision_tree.fit(training_data, training_label)

    def sampling(self, tree: PrimitiveTree):
        # random sampling a non-root node
        index = random_non_leaf_node(tree)
        subtree = tree.searchSubtree(index)
        tuples = parse_tree_to_tuples(tree[subtree])
        self.decision_tree.fit(tuples[0])


def generate_random_tree():
    # 创建一个基础类型"FitnessMin"和一个"Individual"类型
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    pset = gp.PrimitiveSet("MAIN", arity=1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    # 生成一个随机的PrimitiveTree实例
    individual = toolbox.individual()

    # 打印原始树形结构
    print(individual)
    return individual


if __name__ == '__main__':
    tree = generate_random_tree()
    print(parse_tree_to_tuples(tree))
