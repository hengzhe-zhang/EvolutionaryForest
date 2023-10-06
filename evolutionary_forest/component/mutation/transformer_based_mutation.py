import random

import numpy as np
import torch
from deap import gp, base
from scipy.stats import pearsonr
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter

from evolutionary_forest.component.evaluation import single_tree_evaluation
from evolutionary_forest.component.primitive_functions import analytical_quotient
from evolutionary_forest.utils import reset_random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

reset_random(0)


class MutationTransformer(nn.Module):
    def __init__(
        self,
        embedding_class=16,
        embedding_size=64,
        dim_feedforward=64,
        num_layers=2,
        nhead=1,
        dropout=0.1,
        padding_idx=None):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=nhead, dropout=dropout,
                                                   dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding_class = embedding_class
        self.embedding = nn.Embedding(embedding_class, embedding_size, padding_idx=padding_idx)

    def forward(self, batch_x, batch_context):
        torch_input_embedding = self.embedding(batch_x)
        torch_input_embedding = torch.cat([torch_input_embedding, batch_context], dim=1)
        torch_output = self.transformer_encoder.forward(torch.swapdims(torch_input_embedding, 0, 1))
        candidate_primitives = self.embedding(
            torch.IntTensor([i for i in range(self.embedding_class)]).to(device)
        )
        torch_output = torch_output[:number_of_primitives, :, :] @ candidate_primitives.T
        return torch.swapdims(torch_output, 0, 1)


"""
add(a,b)->multiply(a,b)
sin(a)->cos(a)
"""


def tuple_to_individual(pset, t):
    result = []
    for node in t:
        result.append(pset.mapping[node])
    tree = gp.PrimitiveTree(result)
    return tree


def model_training(full_dataset, X,
                   parameters=None):
    pset, primitive_encoder = get_primitive_set(X)
    writer = SummaryWriter('result')

    # Split training data
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    training_data, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # training_data = full_dataset[test_size:]
    # test_dataset = full_dataset[:test_size]
    torch_input, torch_context, torch_target = training_data_process(training_data, primitive_encoder)
    torch_input_test, torch_context_test, torch_target_test = training_data_process(test_dataset, primitive_encoder)
    number_of_primitives = len(primitive_encoder.classes_)
    model = MutationTransformer(embedding_class=number_of_primitives + 1,
                                padding_idx=number_of_primitives,
                                **parameters).to(device)
    loss_fn = nn.CrossEntropyLoss()
    for iter in range(20000):
        # X is a torch Variable
        permutation = torch.randperm(torch_input.size()[0]).to(device)
        batch_size = 128

        # Training in a mini-batch mode
        for i in range(0, torch_input.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = torch_input[indices], torch_target[indices]
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad()
            torch_output = model.forward(batch_x, torch_context[indices])
            loss = loss_fn(torch.swapdims(torch_output, 1, 2), batch_y)
            loss.backward()
            optimizer.step()
        # Evaluate on the training set
        model.eval()
        torch_output_train = model.forward(torch_input, torch_context)
        training_accuracy_nn, training_accuracy_random = calculate_accuracy(number_of_primitives,
                                                                            pset,
                                                                            training_data,
                                                                            torch_input,
                                                                            torch_output_train,
                                                                            torch_context,
                                                                            torch_target,
                                                                            primitive_encoder,
                                                                            testing_flag=False)
        # Evaluate on the test set
        torch_output_test = model.forward(torch_input_test, torch_context_test)
        testing_accuracy_nn, testing_accuracy_random = calculate_accuracy(number_of_primitives,
                                                                          pset,
                                                                          test_dataset,
                                                                          torch_input_test,
                                                                          torch_output_test,
                                                                          torch_context_test,
                                                                          torch_target_test,
                                                                          primitive_encoder)
        model.train()

        writer.add_scalar('Accuracy/Train-Transformer', training_accuracy_nn, iter)
        writer.add_scalar('Accuracy/Train-Random', training_accuracy_random, iter)
        writer.add_scalar('Accuracy/Val-Transformer', testing_accuracy_nn, iter)
        writer.add_scalar('Accuracy/Val-Random', testing_accuracy_random, iter)

    writer.close()
    return training_accuracy, testing_accuracy


def calculate_accuracy(number_of_primitives,
                       pset,
                       all_test_data,
                       torch_input_test,
                       torch_output_test,
                       torch_context_test,
                       torch_target_test,
                       primitive_encoder,
                       testing_flag=True):
    all_inds = []
    for t in all_test_data:
        ind = tuple_to_individual(pset, t[0])
        all_inds.append(ind)

    context_data = torch_context_test.detach().numpy()
    context_y = context_data[:, 0]
    context_x = context_data[:, 1:]

    # Perform mutation
    valid_mutation = 0
    valid_random_mutation = 0
    all_mutation = 0
    label = np.argmax(torch_output_test.detach().numpy(), axis=2)
    # label = torch_target_test.detach().numpy()
    for id, item in enumerate(label):
        ind = all_inds[id]
        # Make the new individual feasible
        current_terminal_id = 0
        for node_id, a in enumerate(ind):
            if isinstance(a, gp.Terminal):
                # do replacement
                ind[node_id] = pset.mapping[f'ARG{current_terminal_id}']
                current_terminal_id += 1
        old_pearson = pearsonr(single_tree_evaluation(ind, pset, context_x[id].T), context_y[id])[0]

        # Try to perform Transformer-based Mutation
        current_pos = 0
        new_nodes = []
        for node_id, a in enumerate(ind):
            if isinstance(a, gp.Primitive):
                # do replacement
                if item[current_pos] < len(primitive_encoder.classes_):
                    # ensure it is a valid replacement
                    new_node = pset.mapping[primitive_encoder.classes_[item[current_pos]]]
                    new_nodes.append(new_node)
                current_pos += 1
            else:
                new_nodes.append(a)
        ind = gp.PrimitiveTree(new_nodes)

        # Evaluate the performance of new features
        new_pearson = pearsonr(single_tree_evaluation(ind, pset, context_x[id].T), context_y[id])[0]
        if np.abs(new_pearson) > 10 * np.abs(old_pearson):
            valid_mutation += 1

        # Try to perform Random Mutation
        new_nodes = []
        for node_id, a in enumerate(ind):
            if isinstance(a, gp.Primitive):
                # ensure it is a valid replacement
                all_primitives = list(filter(lambda x: x.arity == a.arity, pset.primitives[object]))
                new_node = random.choice(all_primitives)
                # print('Old vs New', ind[node_id].name, new_node.name)
                new_nodes.append(new_node)
            else:
                new_nodes.append(a)
        ind = gp.PrimitiveTree(new_nodes)

        # Evaluate the performance of new features
        new_pearson = pearsonr(single_tree_evaluation(ind, pset, context_x[id].T), context_y[id])[0]
        if np.abs(new_pearson) > 10 * np.abs(old_pearson):
            valid_random_mutation += 1
        all_mutation += 1
    print('Accuracy of Mutation', valid_mutation / all_mutation)
    print('Accuracy of Random Mutation', valid_random_mutation / all_mutation)

    for c in range(torch_output_test.shape[1]):
        # Number of different predictions
        testing = 'Testing' if testing_flag else 'Training'
        print(
            'Position', c,
            '%s Accuracy' % testing,
            torch.sum((torch.argmax(torch_output_test[:, c], dim=1) == torch_target_test[:, c])
                      [torch_target_test[:, c] != number_of_primitives]) /
            torch.sum(torch_target_test[:, c] != number_of_primitives)
        )
    total_accuracy = torch.sum((torch.argmax(torch_output_test, dim=2) == torch_target_test)
                               [torch_target_test != number_of_primitives]) / \
                     torch.sum(torch_target_test != number_of_primitives)

    return valid_mutation / all_mutation, valid_random_mutation / all_mutation


def training_data_process(training_data, primitive_encoder):
    # Preparing training data
    torch_input = []
    torch_target = []
    torch_context = []
    for t in training_data:
        t_a = list(filter(lambda x: 'ARG' not in x, t[0]))
        t_b = list(filter(lambda x: 'ARG' not in x, t[1]))
        s_a = primitive_encoder.transform(t_a)
        s_b = primitive_encoder.transform(t_b)
        torch_input.append(torch.IntTensor(s_a))
        context = torch.FloatTensor(t[2])
        if number_of_terminals + 1 - context.shape[1] > 0:
            context = torch.cat([context,
                                 torch.zeros((context.shape[0], number_of_terminals + 1 - context.shape[1]))],
                                dim=1)
        torch_context.append(context.T)
        torch_target.append(torch.LongTensor(s_b))
    torch_input = pad_sequence(torch_input, padding_value=len(primitive_encoder.classes_)).T
    torch_target = pad_sequence(torch_target, padding_value=len(primitive_encoder.classes_)).T
    torch_context = torch.stack(torch_context)

    # input = torch_context.numpy()
    # np.concatenate([torch_input.numpy(), torch_target.numpy()], axis=1)
    # lgbm_score = cross_validate(LGBMClassifier(), input.reshape(input.shape[0], -1)[:, :80],
    #                             torch_target.numpy()[:, 0],
    #                             return_train_score=True)
    # print(lgbm_score['train_score'])
    # print(lgbm_score['test_score'])

    return torch_input.to(device), torch_context.to(device), torch_target.to(device)


def get_primitive_set(X):
    pset = gp.PrimitiveSet("MAIN", X.shape[1])
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(analytical_quotient, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    # pset.addPrimitive(np.sin, 1)
    # pset.addPrimitive(np.cos, 1)
    # pset.addPrimitive(_protected_sqrt, 1)
    primitive_encoder = LabelEncoder()
    primitive_encoder.fit([p.name for p in filter(lambda x: isinstance(x, gp.Primitive), pset.primitives[object])])
    return pset, primitive_encoder


def training_data_generation(X, y):
    # Generating training data
    pset, _ = get_primitive_set(X)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=height)
    training_data = []
    hash_list = set()
    for _ in range(50000):
        id = np.random.randint(0, len(X), embedding_size)
        y_sample = y[id]
        tree = toolbox.expr()

        result_a = single_tree_evaluation(tree, pset, X[id])
        tree_a = tuple(a.name for a in tree)

        # ensure the training data is useful
        while True:
            for node_id, node in enumerate(tree):
                if isinstance(node, gp.Primitive):
                    all_primitives = list(filter(lambda x: x.arity == node.arity, pset.primitives[object]))
                    new_node = random.choice(all_primitives)
                    tree[node_id] = new_node
            tree_b = tuple(a.name for a in tree)
            if tree_b != tree_a:
                break

        used_variable = []
        for node in tree:
            if isinstance(node, gp.Terminal):
                used_variable.append(int(node.name.replace('ARG', '')))
        assert len(used_variable) <= number_of_terminals
        result_b = single_tree_evaluation(tree, pset, X[id])
        used_data = X[id][:, list(used_variable)]
        assert used_data.shape[1] <= number_of_terminals
        used_data = np.concatenate((np.reshape(y[id], (-1, 1)), used_data), axis=1)
        if np.abs(pearsonr(result_b, y_sample)[0]) > 10 * np.abs(pearsonr(result_a, y_sample)[0]):
            # cross_validate(LinearRegression(), np.reshape(result_b, (-1, 1)), y_sample, return_train_score=True)
            # Avoid ambiguous label
            # hash_result = hash(tree_a)
            # if hash_result in hash_list:
            #     continue
            # else:
            #     hash_list.add(hash_result)
            # pearsonr(np.subtract(used_data[:,1],used_data[:,2]),used_data[:,0])[0]
            print(np.abs(pearsonr(result_a, y_sample)[0]), np.abs(pearsonr(result_b, y_sample)[0]))
            training_data.append((tree_a, tree_b, used_data))
        elif np.abs(pearsonr(result_a, y_sample)[0]) > 10 * np.abs(pearsonr(result_b, y_sample)[0]):
            print(np.abs(pearsonr(result_a, y_sample)[0]), np.abs(pearsonr(result_b, y_sample)[0]))
            training_data.append((tree_b, tree_a, used_data))
        else:
            pass
    print('Useful Training Data', len(training_data))
    return training_data


def get_dataset_simple():
    X, y = load_diabetes(return_X_y=True)
    # X, y, _ = get_dataset({
    #     'dataset': '604_fri_c4_500_10'
    # })
    # X, y = load_diabetes(return_X_y=True)
    X, y = np.array(X), np.array(y)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()
    return X, y


if __name__ == '__main__':
    embedding_size = 64
    dim_feedforward = 32
    num_layers = 2
    nhead = 4
    dropout = 0.5
    height = 2
    number_of_primitives = 2 ** height - 1
    number_of_terminals = 2 ** height

    X, y = get_dataset_simple()
    training_data = training_data_generation(X, y)
    training_accuracy, testing_accuracy = model_training(training_data, X,
                                                         {
                                                             'embedding_size': embedding_size,
                                                             'dim_feedforward': dim_feedforward,
                                                             'num_layers': num_layers,
                                                             'nhead': nhead,
                                                             'dropout': dropout,
                                                         })
