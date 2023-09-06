import copy
import operator
import random
from inspect import isclass
from typing import List

import numpy as np
import pandas as pd
from deap import gp, base, creator, tools
from deap.gp import PrimitiveTree, Primitive, Terminal, PrimitiveSet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier


def parse_tree_to_tuples(root: PrimitiveTree):
    stack = []
    tuples = []

    for node in root:
        children = []
        stack.append((node, children))

        while len(stack[-1][1]) == stack[-1][0].arity:
            parent, child_list = stack.pop()
            if len(child_list) > 0:
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


class BuildingBlockLearning():
    def __init__(self, pset):
        # generate a primitive mapping dict
        self.pset: PrimitiveSet = pset

        self.decision_tree = Pipeline([
            ('OneHot', OneHotEncoder(handle_unknown='ignore')),
            ('DT', DecisionTreeClassifier()),
        ])
        self.label_encoder = LabelEncoder()

    def fit(self, trees: List[PrimitiveTree]):
        training_data = []
        training_label = []
        for g in trees:
            single_training_data, single_training_label = self.fit_tree(g)
            training_data.extend(single_training_data)
            training_label.extend(single_training_label)
        training_data = pd.DataFrame(training_data)
        training_label = pd.Series(training_label)
        training_data = training_data.astype('str')
        training_label = training_label.astype('str')
        self.decision_tree.fit(training_data, training_label)
        return self

    def fit_tree(self, g):
        tuples = parse_tree_to_tuples(g)
        training_data = []
        training_label = []
        for tp in tuples:
            for i in range(1, len(tp)):
                # for whole competition
                training_data.append([get_name(t) for t in tp[:i]] +
                                     ['?' for _ in tp[i:]])
                training_label.append(get_name(tp[i]))

                # for point completion
                training_data.append([get_name(t) for t in tp[:i]] + ['?'] +
                                     [get_name(t) for t in tp[i + 1:]])
                training_label.append(get_name(tp[i]))
        return training_data, training_label

    def sampling(self, tree: PrimitiveTree):
        tree = copy.copy(tree)
        # random sampling a non-root node
        index = random_non_leaf_node(tree)
        subtree = tree.searchSubtree(index)
        tuples = parse_tree_to_tuples(tree[subtree])
        current_tree = [get_name(node) for node in tuples[0]]

        # randomly mask one element
        idx = random.randint(1, len(current_tree) - 1)
        current_tree[idx] = '?'

        prediction = self.make_prediction_by_incomplete_tree(current_tree)

        if prediction not in self.pset.mapping:
            # must be random constant
            tree[index + idx] = Terminal(float(prediction), False, float(prediction))
        else:
            # if primitive, recursively generate a whole tree
            if isinstance(self.pset.mapping[prediction], Primitive):
                root_node = self.pset.mapping[prediction]
                subtree_list = [root_node]

                max_height = 3

                def recursive(depth, parent: Primitive):
                    # given existing parent and sibling, generate one node
                    if max_height == depth:
                        only_terminal = True
                    else:
                        only_terminal = False

                    existing = [get_name(parent)]
                    for _ in range(len(parent.args)):
                        now = existing + ['?' for _ in range(parent.arity - (len(existing) - 1))]
                        prediction = self.make_prediction_by_incomplete_tree(now, only_terminal)

                        node = self.pset.mapping[prediction]
                        if only_terminal:
                            assert isinstance(node, Terminal)
                        existing.append(get_name(node))
                        # add node to list
                        if isinstance(node, Terminal):
                            if isclass(node):
                                node = node()
                            subtree_list.append(node)
                        elif isinstance(node, Primitive):
                            subtree_list.append(node)
                            recursive(depth + 1, parent)

                recursive(0, root_node)
            else:
                assert isinstance(self.pset.mapping[prediction], Terminal)
                subtree_list = [self.pset.mapping[prediction]]

            if isinstance(tree[index + idx], Primitive):
                # the original is a subtree
                sub_tree_range = tree.searchSubtree(index + idx)
                tree[sub_tree_range] = subtree_list
            else:
                # the original is a leaf node
                try:
                    tree[slice(index + idx, index + idx + 1)] = subtree_list
                except:
                    pass

        return tree

    def make_prediction_by_incomplete_tree(self, current_tree, only_terminal=False):
        # make predictions
        current_tree = pd.DataFrame([current_tree]).astype(str)
        prediction = self.decision_tree.predict_proba(current_tree)
        prediction = prediction[0]
        if only_terminal:
            possible_index = []
            # only consider terminal variables
            for x in range(len(prediction)):
                if not isinstance(self.pset.mapping[self.decision_tree.classes_[x]], Primitive):
                    possible_index.append(x)
            prediction = prediction[np.array(possible_index)]
            # probability normalization
            prediction = prediction / np.sum(prediction)
        else:
            possible_index = np.arange(0, len(prediction))
        random_id = np.random.choice(possible_index, size=1, p=prediction)
        classes = list(self.decision_tree.classes_)
        prediction = str(classes[random_id[0]])
        return prediction


def generate_random_tree(pset: PrimitiveSet):
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    individual = toolbox.individual()
    print(individual)
    return individual


def pset_intialization():
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    pset = gp.PrimitiveSet("MAIN", arity=1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)
    return pset


if __name__ == '__main__':
    pset = pset_intialization()
    tree = generate_random_tree(pset)
    bl = BuildingBlockLearning(pset)
    bl.fit([generate_random_tree(pset) for _ in range(100)])
    for _ in range(100):
        tree = generate_random_tree(pset)
        print('Original Tree', tree)
        for _ in range(100):
            tree = bl.sampling(tree)
        print('Mutated Tree', tree)
