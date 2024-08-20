import random

import deap.base as base
import deap.creator as creator
import deap.tools as tools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap.gp import PrimitiveSet
from torch.utils.data import DataLoader, TensorDataset


def extract_targets(train_data, pset):
    """
    Extract targets from GP trees in the training data.

    :param train_data: List of tuples (gp_tree, target_semantics) from generate_synthetic_data.
    :param pset: DEAP PrimitiveSet used to decode GP trees.
    :return: List of target tensors.
    """
    targets = []

    # Get the mapping of node names to indices
    node_name_to_index = get_node_name_to_index(pset)

    for tree, _ in train_data:
        tree: creator.Individual

        # Extract node names and convert to indices
        tree_indices = []
        for node in tree:
            node_name = node.name
            if node_name in node_name_to_index:
                tree_indices.append(node_name_to_index[node_name])
            else:
                raise ValueError(
                    f"Node name '{node_name}' not found in node_name_to_index"
                )

        # Convert indices to tensor and add to targets
        target_tensor = torch.tensor(tree_indices, dtype=torch.long)
        targets.append(target_tensor)

    return targets


def get_node_name_to_index(pset):
    """
    Create a mapping from node names to indices based on the PrimitiveSet.

    :param pset: DEAP PrimitiveSet used to decode GP trees.
    :return: Dictionary mapping node names to indices.
    """
    functions, terminals = extract_pset_elements(pset)
    node_names = functions + terminals
    node_name_to_index = {name: idx for idx, name in enumerate(node_names)}
    return node_name_to_index


class NeuralSemanticLibrary(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.1,
        num_symbols=10,
        output_sequence_length=7,
    ):
        super(NeuralSemanticLibrary, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_symbols = num_symbols
        self.output_sequence_length = output_sequence_length

        # Define MLP layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, output_sequence_length * output_size))
        self.mlp = nn.Sequential(*layers)

        # Define the embedding layer
        self.embedding = nn.Embedding(num_symbols, output_size)

    def forward(self, x):
        # Ensure x has shape (batch_size, input_size)
        x = self.mlp(x)

        # Reshape the output to (batch_size, output_sequence_length, output_size)
        x = x.view(-1, self.output_sequence_length, self.output_size)

        # Compute dot product with embedding vectors
        embedding_vectors = self.embedding.weight  # Shape: (num_symbols, output_size)
        # Compute dot product between each element of the sequence and the embedding vectors
        dot_products = torch.matmul(
            x, embedding_vectors.T
        )  # Shape: (batch_size, output_sequence_length, num_symbols)

        return dot_products

    def generate_gp_tree(self, semantics, pset):
        # Generate a GP tree from semantics using DEAP PrimitiveSet
        semantics_tensor = torch.tensor(
            semantics, dtype=torch.float32
        )  # No batch dimension
        with torch.no_grad():
            output_vector = self.forward(semantics_tensor)
            # Get the index of the maximum value along the num_symbols dimension for each time step
            output_indices = torch.argmax(
                output_vector, dim=2
            ).numpy()  # Shape: (output_sequence_length, )
        return decode_to_gp_tree(output_indices, pset)

    def train(self, train_data, batch_size=32, epochs=10, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        tensors = [torch.tensor(d[1], dtype=torch.float32) for d in train_data]
        targets = extract_targets(train_data, pset)
        targets = [
            torch.tensor(t, dtype=torch.long) for t in targets
        ]  # Convert targets to LongTensor

        # Stack tensors and targets to create a dataset
        stacked_tensors = torch.stack(tensors)  # Shape: (num_samples, input_size)
        # Reshape targets to (num_samples, output_sequence_length) if needed
        stacked_targets = torch.stack(targets).view(
            -1, self.output_sequence_length
        )  # Shape: (num_samples, output_sequence_length)

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(stacked_tensors, stacked_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                # Ensure batch_x has shape (batch_size, input_size)
                output = self.forward(batch_x)
                # Flatten the targets to match the output shape for CrossEntropyLoss
                batch_y = batch_y.view(
                    -1
                )  # Shape: (batch_size * output_sequence_length)
                # Flatten the output to match the targets
                output = output.view(
                    -1, self.num_symbols
                )  # Shape: (batch_size * output_sequence_length, num_symbols)
                loss = criterion(output, batch_y)  # No need to flatten
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


def decode_to_gp_tree(indices, pset):
    # Decode indices to a GP tree
    functions, terminals = extract_pset_elements(pset)

    # Example of decoding indices to functions and terminals
    decoded_tree = []
    for idx in indices:
        if idx < len(functions):
            decoded_tree.append(functions[idx])
        else:
            decoded_tree.append(terminals[idx - len(functions)])

    return create_gp_tree(decoded_tree, pset)


def extract_pset_elements(pset):
    # Extract functions and terminals from the DEAP PrimitiveSet
    functions = [prim.name for prim in pset.primitives[object]]
    terminals = [term.name for term in pset.terminals[object]]
    return functions, terminals


def create_gp_tree(decoded_tree, pset):
    # Construct GP tree using DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def generate_function(name):
        return next(f for f in pset.primitives[pset.context] if f.name == name)

    def generate_terminal(name):
        return next(t for t in pset.terminals[pset.context] if t.name == name)

    # Create individual
    def individual():
        tree = [
            generate_function(fn) if isinstance(fn, str) else generate_terminal(fn)
            for fn in decoded_tree
        ]
        return creator.Individual(tree)

    toolbox.register("individual", tools.initIterate, creator.Individual, individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate", lambda x: (1.0,) if isinstance(x, list) else (0.0,)
    )  # Dummy evaluation function

    # Create and return an individual (GP tree)
    return toolbox.individual()


def generate_synthetic_data(pset, num_samples=100):
    """
    Generate synthetic training data with random GP trees and their target semantics.

    :param pset: DEAP PrimitiveSet.
    :param num_samples: Number of samples to generate.
    :return: List of tuples (gp_tree, target_semantics).
    """
    data = []
    for _ in range(num_samples):
        # Generate a random GP tree
        gp_tree = create_random_gp_tree(pset)

        # Generate synthetic semantics (e.g., applying the GP tree to a synthetic dataset)
        target_semantics = generate_target_semantics(gp_tree)

        data.append((gp_tree, target_semantics))
    return data


def create_random_gp_tree(pset):
    # Create a random GP tree
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_func", random.choice, pset.primitives[object])
    toolbox.register("attr_term", random.choice, pset.terminals[object])

    def individual():
        tree = [toolbox.attr_func() for _ in range(3)] + [
            toolbox.attr_term() for _ in range(4)
        ]
        return creator.Individual(tree)

    toolbox.register("individual", tools.initIterate, creator.Individual, individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox.individual()


def generate_target_semantics(gp_tree):
    # Generate synthetic target semantics based on GP tree
    return np.random.randn(10).tolist()  # Example of random semantics


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


if __name__ == "__main__":
    # Example usage
    input_size = 10  # Must be divisible by num_heads
    hidden_size = 64
    output_size = 20  # Number of symbols

    nl = NeuralSemanticLibrary(
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0,
    )

    # Define DEAP PrimitiveSet
    pset = PrimitiveSet("MAIN", 2)  # Example, replace with your specific pset
    pset.addPrimitive(add, 2)
    pset.addPrimitive(subtract, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)

    # Generate synthetic training data
    train_data = generate_synthetic_data(pset, num_samples=100)

    # Train the neural network
    nl.train(train_data, epochs=10, lr=0.001)
