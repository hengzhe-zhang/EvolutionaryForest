import random
import sys
from inspect import isclass

import numpy as np
from deap.tools import HallOfFame


def data_to_tensor(data: np.ndarray, sample: tuple):
    indices = np.random.randint(0, len(data), sample)
    return indices, data[indices]


def hof_to_text(hof: HallOfFame, indices):
    case_values = np.array([x.case_values for x in hof])
    text = []
    for ix in indices:
        # only select top-1 feature
        ind = hof[np.argmin(np.sum(case_values[:, ix], axis=1))]
        index = np.argmax(ind.coef)
        x = ind.gene[index]
        s = ' '.join(tuple(a.name for a in x))
        text.append(s)
    return text


def expr_generate(feature: list, pset, type_=None):
    feature = list(filter(lambda x: x not in ['<unk>', '<pad>', '<bos>', '<eos>'], feature))
    expr = []
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if len(feature) > 0:
            x = feature.pop(0)
            if x in pset.mapping:
                term = pset.mapping[x]
                # constant terminals
                if isclass(term):
                    term = term()
                # primitives
                if hasattr(term, 'args'):
                    for arg in reversed(term.args):
                        stack.append((depth + 1, arg))
                expr.append(term)
            else:
                # constant terminals
                term = pset.terminals[object][-1]()
                term.value = int(x)
                term.name = str(term.value)
        else:
            # add some end terminals
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add " \
                                 "a terminal of type '%s', but there is " \
                                 "none available." % (type_,)).with_traceback(traceback)
            if isclass(term):
                term = term()
            expr.append(term)
    return expr


class TransformerTool():
    def __init__(self, X, y, hof, pset):
        self.transformer = None
        self.pset = pset
        self.X = np.concatenate([X, y.reshape(-1, 1)], axis=1)
        self.hof = hof
        self.embedding_size = 32

    def train(self):
        from .translation_transformer import TranslationTransformer
        from torchtext.data.datasets_utils import _RawTextIterableDataset
        X, hof = self.X, self.hof
        indices, X = data_to_tensor(X, sample=(len(hof), self.embedding_size))
        X = np.swapaxes(X, 1, 2).astype(np.float32)
        text = hof_to_text(hof, indices)

        def data_iterator(split='train'):
            iterator = _RawTextIterableDataset('GP', len(hof), zip(list(X).__iter__(), text.__iter__()))
            return iterator

        epochs = 500
        if self.transformer == None:
            t = TranslationTransformer(data_iterator, EMB_SIZE=self.embedding_size)
            t.train(epochs)
            self.transformer = t
        else:
            t = self.transformer
            t.get_data_iterator = data_iterator
            t.train(epochs)

    def sample(self, num):
        X = self.X
        _, X_test = data_to_tensor(X, sample=(num, self.embedding_size))
        X_test = np.swapaxes(X_test, 1, 2).astype(np.float32)
        samples = [expr_generate(self.transformer.generate(np.expand_dims(x, axis=1)), self.pset, object) for x in
                   X_test]
        return samples
