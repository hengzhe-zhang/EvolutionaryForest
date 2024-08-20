import deap.base as base
import deap.creator as creator
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deap.gp import PrimitiveSet, PrimitiveTree, Primitive
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from evolutionary_forest.probability_gp import genHalfAndHalf


class PrimitiveSetUtils:
    def __init__(self, pset):
        """
        Initialize the utility class with the DEAP PrimitiveSet.

        :param pset: DEAP PrimitiveSet used to decode GP trees.
        """
        self.pset = pset
        self.node_name_to_index = self.get_node_name_to_index()
        self.index_to_node_name = self.get_index_to_node_name()

    def get_node_name_to_index(self):
        """
        Create a mapping from node names to indices based on the PrimitiveSet.

        :return: Dictionary mapping node names to indices.
        """
        functions, terminals = self.extract_pset_elements()
        node_names = [func.name for func in functions] + [
            term.name for term in terminals
        ]
        return {name: idx for idx, name in enumerate(node_names)}

    def get_index_to_node_name(self):
        """
        Create a reverse mapping from indices to node names based on the PrimitiveSet.

        :return: Dictionary mapping indices to node names.
        """
        return {index: name for name, index in self.node_name_to_index.items()}

    def decode_to_functions_and_terminals(self, indices):
        """
        Decode a list of indices into functions and terminals based on the PrimitiveSet.

        :param indices: List of integer indices corresponding to functions and terminals.
        :return: Tuple of lists (functions, terminals) where each list contains the decoded functions and terminals.
        """
        functions = []
        terminals = []

        for idx in indices:
            node_name = self.index_to_node_name.get(idx)
            if node_name is None:
                raise ValueError(f"Index '{idx}' not found in index_to_node_name")

            if node_name in self.pset.primitives:
                functions.append(self.pset.primitives[node_name])
            elif node_name in self.pset.terminals:
                terminals.append(self.pset.terminals[node_name])
            else:
                raise ValueError(f"Node name '{node_name}' not found in PrimitiveSet")

        return functions, terminals

    def extract_targets(self, train_data, max_length):
        """
        Extract targets from GP trees in the training data and pad them to have the same length.

        :param train_data: List of tuples (gp_tree, target_semantics) from generate_synthetic_data.
        :return: List of padded target tensors.
        """
        targets = []

        # First pass: Extract tree indices and determine max length
        tree_indices_list = []
        for tree, _ in train_data:
            tree: creator.Individual

            # Extract node names and convert to indices
            tree_indices = []
            for node in tree:
                node_name = node.name
                if node_name in self.node_name_to_index:
                    tree_indices.append(self.node_name_to_index[node_name])
                else:
                    raise ValueError(
                        f"Node name '{node_name}' not found in node_name_to_index"
                    )

            # Store the tree indices and update max_length
            tree_indices_list.append(tree_indices)

        # Second pass: Pad tree indices and convert to tensors
        for tree_indices in tree_indices_list:
            # Pad the indices
            padded_indices = tree_indices + [0] * (max_length - len(tree_indices))

            # Convert to tensor and add to targets
            target_tensor = torch.tensor(padded_indices, dtype=torch.long)
            targets.append(target_tensor)

        return targets

    def extract_pset_elements(self):
        """
        Extract function and terminal names from the PrimitiveSet.

        :return: Tuple of lists (functions, terminals), where each list contains names of elements.
        """
        functions = self.pset.primitives[object]
        terminals = self.pset.terminals[object]
        return functions, terminals

    def convert_node_names_to_gp_tree(self, node_names):
        """
        Convert a list of node names into a DEAP GP tree.

        :param node_names: List of node names corresponding to functions and terminals.
        :return: DEAP GP tree constructed from the node names.
        """
        # Create a dictionary to map function names to their DEAP functions
        func_dict = {func.name: func for func in self.pset.primitives[object]}
        term_dict = {term.name: term for term in self.pset.terminals[object]}

        def build_tree(node_names):
            track = 0
            if not node_names:
                return None

            stack = []
            for idx, name in enumerate(node_names):
                if idx > 0 and track == 0:
                    break
                if name in func_dict:
                    func = func_dict[name]
                    arity = func.arity
                    if track != 0:
                        track -= 1
                    track += arity
                    stack.append(func_dict[name])
                elif name in term_dict:
                    track -= 1
                    terminal = term_dict[name]
                    if callable(terminal):
                        terminal = terminal()
                    stack.append(terminal)
                else:
                    raise ValueError(f"Node name '{name}' not found in PrimitiveSet")
            return stack

        def reorder_node_names_recursive(node_names):
            """
            Reorder a list of node names such that each function is followed by its arguments using recursion.

            :param node_names: List of node names corresponding to functions and terminals.
            :return: Reordered list of node names.
            """
            if not node_names:
                return []

            # Take the first node from the list
            first = node_names.pop(0)

            # If it's a function (Primitive), it will have arguments
            if isinstance(first, Primitive):
                # Recursively get the reordered arguments based on the arity of the function
                args = [
                    reorder_node_names_recursive(node_names) for _ in range(first.arity)
                ]
                # Flatten the list of arguments and append the function at the beginning
                return [first] + [item for sublist in args for item in sublist]
            else:
                # If it's a terminal, just return it
                return [first]

        return reorder_node_names_recursive(build_tree(node_names))


def get_max_arity(pset):
    """
    Extract the maximum arity of functions from the DEAP PrimitiveSet.

    :param pset: DEAP PrimitiveSet object containing functions and terminals.
    :return: Maximum arity of functions in the PrimitiveSet.
    """
    max_arity = 0

    # Iterate over functions in the PrimitiveSet
    for func in pset.primitives[object]:
        if hasattr(func, "arity"):
            max_arity = max(max_arity, func.arity)

    return max_arity


def calculate_terminals_needed(num_functions, pset):
    """
    Calculate the number of terminals needed to ensure a complete tree in GEP.

    :param num_functions: Number of primitive functions.
    :param max_arity: Maximum arity of the functions.
    :return: Number of terminals required.
    """
    max_arity = get_max_arity(pset)
    if num_functions < 0 or max_arity <= 0:
        raise ValueError(
            "Number of functions must be non-negative and maximum arity must be positive."
        )

    # Calculate the number of terminals required
    num_terminals = 1 + (num_functions * (max_arity - 1))

    return num_functions + num_terminals


class NeuralSemanticLibrary(nn.Module):
    def __init__(
        self,
        input_size=10,
        hidden_size=64,
        output_size=20,
        num_layers=1,
        dropout=0.1,
        output_primitive_length=3,
        pset=None,  # Add pset parameter
    ):
        super(NeuralSemanticLibrary, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_primitive_length = output_primitive_length
        max_nodes = calculate_terminals_needed(output_primitive_length, pset)
        self.output_sequence_length = max_nodes

        self.pset_utils = (
            PrimitiveSetUtils(pset) if pset else None
        )  # Initialize PrimitiveSetUtils
        self.num_symbols = (
            len(self.pset_utils.node_name_to_index) if self.pset_utils else 0
        )  # Dynamically determine number of symbols
        self.num_terminals = (
            len(self.pset_utils.pset.terminals[object]) if self.pset_utils else 0
        )  # Store num_terminals as class attribute

        # Define MLP layers
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, self.output_sequence_length * output_size))
        self.mlp = nn.Sequential(*layers)

        # Define the embedding layer
        self.embedding = nn.Embedding(self.num_symbols, output_size)

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

    def generate_gp_tree(self, semantics):
        number_of_terminals = self.output_sequence_length - self.output_primitive_length

        # Convert semantics to tensor
        semantics_tensor = torch.tensor(semantics, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension

        # Perform forward pass
        with torch.no_grad():
            output_vector = self.forward(semantics_tensor)

            # Create a mask to ensure that the last four positions are terminals
            mask = torch.ones_like(
                output_vector
            )  # Shape: (batch_size, output_sequence_length, num_symbols)

            # Set the mask values for terminals and non-terminals
            mask[:, -number_of_terminals:, :] = 0
            mask[
                :, -number_of_terminals:, -self.num_terminals :
            ] = 1  # Ensure last four positions are terminals

            # Apply the mask to the output vector
            masked_output_vector = output_vector * mask

            # Get the index of the maximum value along the num_symbols dimension for each time step
            output_indices = (
                torch.argmax(masked_output_vector, dim=2).squeeze(0).numpy()
            )  # Shape: (output_sequence_length,)

        # Get the reverse mapping from indices to node names
        index_to_node_name = self.pset_utils.get_index_to_node_name()

        # Decode indices to node names
        tree_node_names = []
        for i, idx in enumerate(output_indices):
            node_name = index_to_node_name.get(idx, "Unknown")
            tree_node_names.append(node_name)

        return tree_node_names

    def train(
        self,
        train_data,
        batch_size=32,
        epochs=1000,
        lr=0.001,
        val_split=0.2,
        patience=20,
        verbose=False,
    ):
        """
        Train the neural network with optional validation and early stopping.

        :param train_data: List of tuples (gp_tree, target_semantics).
        :param batch_size: Batch size for training.
        :param epochs: Number of epochs for training.
        :param lr: Learning rate.
        :param val_split: Fraction of training data to use as validation.
        :param patience: Number of epochs to wait for improvement before early stopping.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # Extract targets and tensors
        tensors = [torch.tensor(d[1], dtype=torch.float32) for d in train_data]
        targets = self.pset_utils.extract_targets(
            train_data, self.output_sequence_length
        )
        targets = [torch.tensor(t, dtype=torch.long) for t in targets]

        # Split data into training and validation sets if val_split > 0
        if val_split > 0:
            train_tensors, val_tensors, train_targets, val_targets = train_test_split(
                tensors, targets, test_size=val_split, random_state=0
            )
            train_data = list(zip(train_tensors, train_targets))
            val_data = list(zip(val_tensors, val_targets))
        else:
            val_data = None

        # Stack tensors and targets to create a dataset
        stacked_tensors = torch.stack(train_tensors)  # Shape: (num_samples, input_size)
        # Reshape targets to (num_samples, output_sequence_length) if needed
        stacked_targets = torch.stack(train_targets).view(
            -1, self.output_sequence_length
        )  # Shape: (num_samples, output_sequence_length)

        # Create TensorDataset and DataLoader
        dataset = TensorDataset(stacked_tensors, stacked_targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            self.mlp.train()  # Set the model to training mode
            for batch_x, batch_y in dataloader:
                # Ensure batch_x has shape (batch_size, input_size)
                output = self.forward(batch_x)

                # Create a mask to ensure that the last four positions are terminals
                mask = torch.ones_like(
                    output
                )  # Shape: (batch_size, output_sequence_length, num_symbols)

                # Apply the mask to the output vector
                masked_output = output * mask

                # Flatten the targets and apply the mask
                batch_y = batch_y.view(
                    -1
                )  # Shape: (batch_size * output_sequence_length)
                # Flatten the masked output to match the targets
                masked_output = masked_output.view(
                    -1, self.num_symbols
                )  # Shape: (batch_size * output_sequence_length, num_symbols)

                # Calculate loss only for masked positions
                loss = criterion(masked_output, batch_y)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(dataloader)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}")

            # Validation
            if val_data:
                self.mlp.eval()  # Set the model to evaluation mode
                val_tensors, val_targets = zip(*val_data)
                val_tensors = torch.stack(val_tensors)
                val_targets = torch.stack(val_targets).view(
                    -1, self.output_sequence_length
                )

                val_output = self.forward(val_tensors)
                val_mask = torch.ones_like(val_output)
                val_masked_output = val_output * val_mask

                val_targets = val_targets.view(-1)
                val_masked_output = val_masked_output.view(-1, self.num_symbols)

                val_loss = criterion(val_masked_output, val_targets).item()
                if verbose:
                    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Optionally save the model here
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(
                                "Early stopping due to no improvement in validation loss."
                            )
                        break

    def convert_to_primitive_tree(self, semantics):
        """
        Convert semantics to a GP tree and then to a PrimitiveTree.

        :param semantics: List of node names representing the GP tree.
        :return: PrimitiveTree constructed from the generated GP tree.
        """
        node_names = self.generate_gp_tree(semantics)
        gp_tree = self.pset_utils.convert_node_names_to_gp_tree(node_names)
        return PrimitiveTree(gp_tree)


def generate_synthetic_data(pset, num_samples=100):
    """
    Generate synthetic training data with random GP trees and their target semantics.

    :param pset: DEAP PrimitiveSet.
    :param num_samples: Number of samples to generate.
    :return: List of tuples (gp_tree, target_semantics).
    """
    # Create a random GP tree using DEAP's ramped half-and-half method
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    data = []
    for _ in range(num_samples):
        # Generate a random GP tree
        gp_tree = create_random_gp_tree(pset)

        # Generate synthetic semantics (e.g., applying the GP tree to a synthetic dataset)
        target_semantics = generate_target_semantics(gp_tree)

        data.append((gp_tree, target_semantics))
    return data


def create_random_gp_tree(pset, min_depth=1, max_depth=2):
    toolbox = base.Toolbox()
    # Initialize ramped half-and-half method
    toolbox.register(
        "individual",
        genHalfAndHalf,
        pset=pset,
        min_=min_depth,
        max_=max_depth,
    )

    return toolbox.individual()


def generate_target_semantics(gp_tree):
    # Generate synthetic target semantics based on GP tree
    return np.random.randn(10).tolist()  # Example of random semantics


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def filter_train_data_by_node_count(train_data, max_nodes=7):
    """
    Filter out GP trees from train_data with more than a specified number of nodes.

    :param train_data: List of tuples (gp_tree, target_semantics), where gp_tree is a DEAP Individual.
    :param max_nodes: Maximum number of nodes allowed in the GP trees.
    :return: List of tuples (gp_tree, target_semantics) with GP trees having nodes count <= max_nodes.
    """
    filtered_train_data = [
        (tree, semantics) for tree, semantics in train_data if len(tree) <= max_nodes
    ]
    return filtered_train_data


if __name__ == "__main__":
    np.random.seed(0)
    # Example usage
    input_size = 10  # Must be divisible by num_heads
    hidden_size = 64
    output_size = 20  # Number of symbols

    # Define DEAP PrimitiveSet
    pset = PrimitiveSet("MAIN", 2)  # Example, replace with your specific pset
    pset.addPrimitive(add, 2)
    pset.addPrimitive(subtract, 2)
    pset.addTerminal(1)
    pset.addTerminal(2)

    nl = NeuralSemanticLibrary(
        input_size, hidden_size, output_size, num_layers=1, dropout=0, pset=pset
    )

    # Generate synthetic training data
    train_data = generate_synthetic_data(pset, num_samples=100)
    train_data = filter_train_data_by_node_count(train_data)

    # Train the neural network
    nl.train(train_data, epochs=100, lr=0.001)
    print(str(nl.convert_to_primitive_tree(train_data[0][1])))
    print(str(PrimitiveTree(train_data[0][0])))
