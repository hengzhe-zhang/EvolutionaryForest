import torch
import torch.nn as nn
import torch.optim as optim
from deap.gp import PrimitiveSet
from torch.nn import Transformer
import numpy as np
import deap.tools as tools
import deap.creator as creator
import deap.base as base
import random


class NeuralSemanticLibrary(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.1,
        num_symbols=20,
    ):
        super(NeuralSemanticLibrary, self).__init__()

        num_heads = 4  # This should divide input_size

        # Ensure input_size is divisible by num_heads
        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"

        self.transformer = Transformer(
            d_model=input_size,  # This must be equal to the feature dimension of src and tgt
            nhead=num_heads,  # Number of attention heads
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_size,
            dropout=dropout,
        )
        self.fc = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(
            num_symbols, input_size
        )  # Embedding layer for decoding

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_symbols = num_symbols

    def forward(self, src, tgt):
        # Ensure src and tgt have shape (sequence_length, batch_size, input_size)
        assert (
            src.size(2) == self.input_size
        ), "src feature dimension must be equal to d_model"
        assert (
            tgt.size(2) == self.input_size
        ), "tgt feature dimension must be equal to d_model"

        transformer_output = self.transformer(src, tgt)
        output = transformer_output[-1, :, :]  # Take the output of the last time step
        output = self.fc(output)
        return output

    def generate_gp_tree(self, semantics, pset):
        # Generate a GP tree from semantics using DEAP PrimitiveSet
        semantics_tensor = torch.tensor(semantics, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension
        src = semantics_tensor
        tgt = semantics_tensor  # Assuming src and tgt are the same here for simplicity
        with torch.no_grad():
            output_vector = self.forward(src, tgt).squeeze(
                0
            )  # (batch_size, output_size)
            output_indices = torch.argmax(
                output_vector, dim=1
            ).numpy()  # Decode to indices
        return decode_to_gp_tree(output_indices, pset)

    def train(self, train_data, epochs=10, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()  # Assuming classification problem

        for epoch in range(epochs):
            total_loss = 0
            for gp_tree, target_semantics in train_data:
                target_tensor = torch.tensor(
                    target_semantics, dtype=torch.long
                ).unsqueeze(
                    0
                )  # Long for classification
                src = target_tensor
                tgt = target_tensor
                output_vector = self.forward(src, tgt)
                loss = criterion(
                    output_vector.view(-1, self.num_symbols), target_tensor.view(-1)
                )  # Flatten for CrossEntropyLoss
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_data)}")


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
    functions = [str(f) for f in pset.primitives if callable(f)]
    terminals = [str(t) for t in pset.terminals]
    return functions, terminals


def create_gp_tree(decoded_tree, pset):
    # Construct GP tree using DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def generate_function(name):
        return next(f for f in pset.primitives if str(f) == name)

    def generate_terminal(name):
        return next(t for t in pset.terminals if str(t) == name)

    # Create individual
    def individual():
        tree = [
            generate_function(fn) if callable(fn) else generate_terminal(fn)
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
    toolbox.register("attr_func", random.choice, pset.primitives)
    toolbox.register("attr_term", random.choice, pset.terminals)

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
    return np.random.randint(0, 20, size=(10,)).tolist()  # Example of random semantics


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


if __name__ == "__main__":
    # Example usage
    input_size = 16  # Must be divisible by num_heads
    hidden_size = 64
    output_size = 20  # Number of symbols
    num_heads = 4

    nl = NeuralSemanticLibrary(
        input_size,
        hidden_size,
        output_size,
        num_layers=1,
        dropout=0.1,
        num_symbols=output_size,
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
    nl.train(train_data, epochs=5, lr=0.001)
