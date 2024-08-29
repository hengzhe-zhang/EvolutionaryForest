import deap.base as base
import deap.creator as creator
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deap import gp
from deap.gp import PrimitiveSet, PrimitiveTree, Primitive
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset

from evolutionary_forest.component.crossover.kan import KANLinear
from evolutionary_forest.probability_gp import genHalfAndHalf
from evolutionary_forest.utility.normalization_tool import normalize_vector
from evolutionary_forest.utility.retrieve_nn.quick_retrive import (
    retrieve_nearest_y,
    retrieve_nearest_y_skip_self,
)
from evolutionary_forest.utility.tree_parsing import mark_node_levels_recursive


def compute_accuracy(val_output, val_targets, padding_idx):
    """
    Compute the accuracy of predictions, ignoring the padding index.

    :param val_output: Tensor of shape (batch_size, seq_len, num_symbols) containing the model outputs.
    :param val_targets: Tensor of shape (batch_size, seq_len) containing the true target values.
    :param padding_idx: Index used for padding in the target tensor.
    :return: Accuracy as a float.
    """
    # Flatten the tensors
    val_targets = val_targets.view(-1)
    val_output = val_output.view(-1, val_output.size(-1))

    # Get the predicted values by taking the argmax over the last dimension
    val_predictions = torch.argmax(val_output, dim=-1)

    # Create a mask that ignores the padding index
    mask = val_targets != padding_idx

    # Calculate the number of correct predictions
    correct_predictions = (val_predictions == val_targets) & mask
    correct_count = correct_predictions.sum().item()

    # Calculate the total number of valid (non-padding) targets
    valid_count = mask.sum().item()

    # Compute the accuracy
    accuracy = correct_count / valid_count if valid_count > 0 else 0.0

    return accuracy


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(RotaryPositionalEmbedding, self).__init__()
        self.dim = d_model // 2

    def forward(self, x):
        seq_len = x.size(1)
        freqs = torch.arange(self.dim, dtype=torch.float32, device=x.device)
        freqs = freqs / self.dim
        freqs = 1.0 / (10000**freqs)
        angles = torch.arange(seq_len, dtype=torch.float32, device=x.device)
        angles = torch.einsum("i,j->ij", angles, freqs)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        encoding = encoding.unsqueeze(0).expand_as(x)
        return x * encoding


def plot_similarity_matrices(
    cosine_similarity_features,
    cosine_similarity_target_features,
    title_features="Cosine Similarity (Features)",
    title_target="Cosine Similarity (Target Features)",
):
    """
    Visualize and compare cosine similarity matrices side by side.

    :param cosine_similarity_features: Cosine similarity matrix computed from features.
    :param cosine_similarity_target_features: Cosine similarity matrix computed from target_features.
    :param title_features: Title for the features similarity matrix plot.
    :param title_target: Title for the target features similarity matrix plot.
    """
    # Convert tensors to numpy arrays for plotting
    cosine_similarity_features = cosine_similarity_features.detach().cpu().numpy()
    cosine_similarity_target_features = (
        cosine_similarity_target_features.detach().cpu().numpy()
    )

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Plot the cosine similarity matrix for features
    sns.heatmap(
        cosine_similarity_features, ax=axes[0], cmap="viridis", cbar=True, annot=False
    )
    axes[0].set_title(title_features)
    axes[0].set_xlabel("Batch Index")
    axes[0].set_ylabel("Batch Index")

    # Plot the cosine similarity matrix for target features
    sns.heatmap(
        cosine_similarity_target_features,
        ax=axes[1],
        cmap="viridis",
        cbar=True,
        annot=False,
    )
    axes[1].set_title(title_target)
    axes[1].set_xlabel("Batch Index")
    axes[1].set_ylabel("Batch Index")

    plt.show()


class DataInputLayer(nn.Module):
    def __init__(self, b, num_weights):
        super(DataInputLayer, self).__init__()
        # Create a linear layer with multiple sets of weights (num_weights) of size b
        self.linear = nn.Linear(b, num_weights)

    def forward(self, x):
        # x should have shape [batch_size, a, b]
        # Apply the linear layer to each line of a (along the last dimension)
        x = self.linear(x)  # Shape: [batch_size, a, num_weights]

        # Apply max pooling along the dimension corresponding to a
        x, _ = torch.max(x, dim=1)  # Shape: [batch_size, num_weights]

        return x


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
        return {
            name: idx + 2 for idx, name in enumerate(node_names)
        }  # Start indexing from 1

    def get_index_to_node_name(self):
        """
        Create a reverse mapping from indices to node names based on the PrimitiveSet.

        :return: Dictionary mapping indices to node names.
        """
        index_to_node_name = {
            1: "<start>",
            0: "<end>",
        }  # Index 0 is reserved for padding and mapped to an empty string or None
        index_to_node_name.update(
            {idx: name for name, idx in self.node_name_to_index.items()}
        )
        return index_to_node_name

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
        padding_index = 0  # Use 0 as the padding index

        # First pass: Extract tree indices and determine max length
        tree_indices_list = []
        for tree, _ in train_data:
            tree: creator.Individual
            tree = self.convert_gp_tree_to_node_names(tree)
            # self.convert_node_names_to_gp_tree([node.name for node in tree])
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
            # Pad the indices with the padding index
            padded_indices = tree_indices + [padding_index] * (
                max_length - len(tree_indices)
            )

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

        def reorder_node_names_recursive(first, node_names):
            """
            Reorder a list of node names such that each function is followed by its arguments using recursion.

            :param node_names: List of node names corresponding to functions and terminals.
            :return: Reordered list of node names.
            """
            # If it's a function (Primitive), it will have arguments
            if isinstance(first, Primitive):
                args = []
                root_args = []
                for ix in range(first.arity):
                    a = node_names.pop(0)
                    # print(first.name, a.name)
                    root_args.append(a)

                for ix, a in enumerate(root_args):
                    nodes = reorder_node_names_recursive(a, node_names)
                    # print(first.name, [n.name for n in nodes])
                    args.extend(nodes)

                # Return the function followed by its arguments
                return [first] + args
            else:
                # If it's a terminal, just return it
                return [first]

        def pre_recursive(node_names):
            first = node_names.pop(0)
            return reorder_node_names_recursive(first, node_names)

        return pre_recursive(build_tree(node_names))

    def convert_gp_tree_to_node_names(self, gp_tree):
        level, _ = mark_node_levels_recursive(gp_tree, original_primitive=True)
        reordered_list = list(map(lambda x: x[0], sorted(level, key=lambda x: x[1])))
        return reordered_list


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
        hidden_size=16,
        output_size=16,
        num_layers=1,
        dropout=0.0,
        output_primitive_length=3,
        pset=None,  # Add pset parameter
        batch_norm=True,  # Add batch_norm parameter
        residual=True,  # Add residual connection parameter
        padding_idx=0,  # Padding index for embedding
        num_heads=4,  # Add number of attention heads
        transformer_layers=2,  # Add number of transformer layers
        use_transformer=True,  # Flag to enable or disable transformer layer
        contrastive_loss_in_val=True,  # Add flag to enable contrastive loss in validation
        flatten_before_similarity=False,  # Add flag to support flatten before calculating similarity
        use_decoder_transformer=True,  # Flag to enable or disable decoder transformer
        contrastive_learning_stage="Decoder",
        selective_retrain=False,
        use_x_transformer=False,  # Add flag to enable PerformerLM
    ):
        super(NeuralSemanticLibrary, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.output_primitive_length = output_primitive_length
        self.batch_norm = batch_norm  # Store batch_norm flag
        self.residual = residual  # Store residual connection flag
        self.use_transformer = use_transformer  # Store use_transformer flag
        self.flatten_before_similarity = flatten_before_similarity  # Store flatten flag
        self.use_decoder_transformer = use_decoder_transformer

        max_nodes = calculate_terminals_needed(output_primitive_length, pset)
        self.output_sequence_length = max_nodes

        self.pset_utils = (
            PrimitiveSetUtils(pset) if pset else None
        )  # Initialize PrimitiveSetUtils
        self.num_symbols = (
            len(self.pset_utils.node_name_to_index) + 2 if self.pset_utils else 0
        )  # Dynamically determine number of symbols (+1 for padding)
        self.num_terminals = (
            len(self.pset_utils.pset.terminals[object]) if self.pset_utils else 0
        )  # Store num_terminals as class attribute

        # Define MLP layers
        layers = []
        for _ in range(num_layers):
            kan = True
            if kan:
                layers.append(KANLinear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(input_size, hidden_size))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            if not kan:
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        self.use_x_transformer = use_x_transformer
        if self.use_x_transformer:
            layers.append(nn.Linear(hidden_size, output_size))
        else:
            layers.append(
                nn.Linear(hidden_size, self.output_sequence_length * output_size)
            )
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()  # Initialize weights

        if transformer_layers == 0:
            self.use_transformer = False

        if self.use_transformer:
            # Transformer Encoder for nearest `y`
            transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=output_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                activation="gelu",
            )
            self.transformer_encoder = nn.TransformerEncoder(
                transformer_encoder_layer, num_layers=transformer_layers
            )

            # Transformer Decoder for combined `x` and encoded nearest `y`
            transformer_decoder_layer = nn.TransformerDecoderLayer(
                d_model=output_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                dropout=dropout,
                activation="gelu",
            )
            self.transformer_decoder = nn.TransformerDecoder(
                transformer_decoder_layer, num_layers=transformer_layers
            )

        # Define a shared embedding layer for all positions
        self.embedding = nn.Embedding(
            self.num_symbols, output_size, padding_idx=padding_idx
        )
        self.output_linear = nn.Linear(output_size, self.num_symbols)

        self.start_token_index = 1
        self.contrastive_loss_in_val = contrastive_loss_in_val
        self.contrastive_learning_stage = contrastive_learning_stage
        self.selective_retrain = selective_retrain

        # Positional embedding initialization
        self.positional_embedding = nn.Embedding(
            self.output_sequence_length, output_size
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, batch_y=None, nearest_y=None):
        x = self._process_mlp(x)
        return self._forward_traditional_transformer(x, batch_y, nearest_y)

    def _forward_x_transformer(self, x, batch_y, nearest_y):
        enc = self.transformer_decoder.encoder(nearest_y, return_embeddings=True)
        enc = torch.cat((x.unsqueeze(1), enc), dim=1)
        output = self.transformer_decoder.decoder.net(batch_y, context=enc)
        return output, output if self.contrastive_learning_stage == "Decoder" else enc

    def _forward_traditional_transformer(self, x, batch_y, nearest_y):
        x = self._reshape_output(x)
        x = self._add_positional_embedding(x)

        if self.use_transformer and nearest_y is not None:
            nearest_y_encoded = self._encode_nearest_y(nearest_y)
            combined_features = self._combine_features(x, nearest_y_encoded)

            if self.use_decoder_transformer:
                output = self._decode(combined_features, batch_y)
            else:
                output = combined_features
        else:
            output = x

        dot_products = self.output_linear(output)
        return (
            dot_products,
            output
            if self.contrastive_learning_stage == "Decoder"
            else nearest_y_encoded,
        )

    def _process_mlp(self, x):
        """Pass the input through MLP layers with optional residual connections."""
        x = self.mlp[0](x)
        residual = x
        for layer in self.mlp[1:-1]:
            if isinstance(layer, nn.Linear) and self.residual:
                x = layer(x) + residual
                residual = x
            else:
                x = layer(x)
        x = self.mlp[-1](x)
        return x

    def _reshape_output(self, x):
        """Reshape the output to (batch_size, output_sequence_length, output_size)."""
        return x.view(-1, self.output_sequence_length, self.output_size)

    def _add_positional_embedding(self, x):
        """Add positional embeddings to the input sequence."""
        batch_size, seq_length, _ = x.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        positional_embeds = self.positional_embedding(position_ids)
        return x + positional_embeds
        # return x

    def _add_positional_embedding_index(self, x, position):
        """Add positional embeddings to the input sequence."""
        position_ids = torch.full(
            (x.size(0), x.size(1)), position, dtype=torch.long, device=x.device
        )
        positional_embeds = self.positional_embedding(position_ids)
        return x + positional_embeds
        # return x

    def _encode_nearest_y(self, nearest_y):
        """Encode nearest `y` using the embedding layer and Transformer Encoder."""
        nearest_y_embedded = self.embedding(nearest_y)
        nearest_y_embedded = self._add_positional_embedding(nearest_y_embedded)
        nearest_y_encoded = self.transformer_encoder(
            nearest_y_embedded.permute(1, 0, 2)
        ).permute(1, 0, 2)
        return nearest_y_encoded

    def _combine_features(self, x, nearest_y_encoded, method="add"):
        """Combine the MLP output `x` and encoded nearest `y`."""
        if method == "concat":
            # Concatenate along the last dimension (feature dimension)
            return torch.cat((x, nearest_y_encoded), dim=1)
        elif method == "add":
            return x + nearest_y_encoded
        else:
            raise ValueError("Unsupported combination method: use 'concat' or 'add'.")

    def _decode(self, combined_features, batch_y):
        """Decode using Transformer Decoder."""
        if batch_y is not None:
            return self._decode_training(combined_features, batch_y)
        else:
            return self._decode_inference(combined_features)

    def _decode_training(self, combined_features, batch_y):
        """Decode in training mode with teacher forcing."""
        start_tokens = torch.full(
            (batch_y.size(0), 1),
            self.start_token_index,
            dtype=torch.long,
            device=combined_features.device,
        )
        batch_y = torch.cat([start_tokens, batch_y[:, :-1]], dim=1)
        tgt = self.embedding(batch_y)
        tgt = self._add_positional_embedding(tgt)
        tgt = tgt.permute(1, 0, 2)
        memory = combined_features.permute(1, 0, 2)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
            tgt.device
        )
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return output.permute(1, 0, 2)

    def _decode_inference(self, combined_features):
        """Decode in inference mode by generating the sequence step-by-step."""
        batch_size = combined_features.size(0)
        tgt = self.embedding(
            torch.full(
                (batch_size, 1),
                self.start_token_index,
                dtype=torch.long,
                device=combined_features.device,
            )
        )
        tgt = self._add_positional_embedding_index(
            tgt, position=0
        )  # Add positional embedding for the first token
        tgt = tgt.permute(1, 0, 2)
        memory = combined_features.permute(1, 0, 2)

        for i in range(1, self.output_sequence_length + 1):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(
                tgt.device
            )
            output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            output = output.permute(1, 0, 2)
            last_token_logits = self.output_linear(output[:, -1:])
            next_token = last_token_logits.argmax(dim=-1)
            if i == self.output_sequence_length:
                break
            next_token_embedding = self.embedding(next_token)
            next_token_embedding = self._add_positional_embedding_index(
                next_token_embedding, position=i
            )  # Add positional embedding for the new token
            next_token_embedding = next_token_embedding.permute(1, 0, 2)
            tgt = torch.cat([tgt, next_token_embedding], dim=0)

        return output

    def _compute_dot_product(self, output):
        """Compute the dot product with shared embedding vectors."""
        embedding_vectors = self.embedding.weight
        return torch.matmul(output, embedding_vectors.T)

    def contrastive_loss(self, features, target_features, margin=0.5):
        """
        Compute contrastive loss to enforce that pairwise cosine distances between instances using features
        should correlate with pairwise cosine distances using target_features.

        :param features: The learned feature vectors with shape [batch_size, sequence_length, feature_dim].
        :param target_features: The original input features with shape [batch_size, input_dim].
        :param margin: Margin for contrastive loss.
        :return: Calculated contrastive loss.
        """
        # Flatten or mean features across sequence length before calculating similarity
        if self.flatten_before_similarity:
            features = features.reshape(features.size(0), -1)  # Flatten features
        else:
            features = features.mean(dim=1)  # Shape: [batch_size, feature_dim]

        # Normalize the feature vectors after mean or flatten
        features = F.normalize(
            features, p=2, dim=-1
        )  # Normalize across the feature_dim
        target_features = F.normalize(
            target_features, p=2, dim=-1
        )  # Normalize across the input_dim

        # Compute pairwise cosine similarity matrices
        cosine_similarity_features = torch.matmul(
            features, features.T
        )  # Shape: [batch_size, batch_size]
        cosine_similarity_target_features = torch.abs(
            torch.matmul(target_features, target_features.T)
        )  # Shape: [batch_size, batch_size]

        # Convert cosine similarity to cosine distance
        cosine_distance_features = 1 - cosine_similarity_features
        cosine_distance_target_features = 1 - cosine_similarity_target_features

        # Calculate the margin-based contrastive loss
        # The margin term penalizes large deviations where the learned distance is larger than the target distance by a margin
        margin_loss = F.relu(
            cosine_distance_features - cosine_distance_target_features + margin
        ).mean()

        # Calculate the loss as the MSE loss combined with the margin loss
        loss = (
            F.mse_loss(cosine_distance_features, cosine_distance_target_features)
            + margin_loss
        )

        return loss

    def train(
        self,
        train_data,
        batch_size=64,
        epochs=1000,
        lr=0.001,
        val_split=0.2,
        patience=20,
        verbose=False,
        loss_weight=0.5,
        min_loss_diff=0.01,
    ):
        optimizer, scheduler = self.setup_optimizer_and_scheduler(lr)
        criterion = nn.CrossEntropyLoss(ignore_index=self.embedding.padding_idx)

        tensors, targets = self.prepare_data(train_data)
        train_tensors, val_tensors, train_targets, val_targets = self.split_data(
            tensors, targets, val_split
        )

        train_tensors, val_tensors = self.normalize_data(train_tensors, val_tensors)
        val_data = self.prepare_validation_data(
            val_tensors, val_targets, train_tensors, train_targets
        )

        if val_data and self.selective_retrain:
            if self.should_skip_training(val_data, loss_weight, verbose):
                return self.previous_val_loss

        dataset, dataloader = self.create_dataloader(
            train_tensors, train_targets, batch_size
        )
        best_val_loss = self.execute_training(
            dataloader,
            criterion,
            optimizer,
            scheduler,
            val_data,
            loss_weight,
            patience,
            epochs,
            verbose,
        )

        self.previous_val_loss = best_val_loss
        return best_val_loss

    def setup_optimizer_and_scheduler(self, lr):
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )
        return optimizer, scheduler

    def prepare_data(self, train_data):
        tensors = [torch.tensor(d[1], dtype=torch.float32) for d in train_data]
        targets = self.pset_utils.extract_targets(
            train_data, self.output_sequence_length
        )
        targets = [torch.tensor(t, dtype=torch.long) for t in targets]
        tensors = torch.stack(tensors)
        return tensors, targets

    def split_data(self, tensors, targets, val_split):
        train_tensors, val_tensors, train_targets, val_targets = train_test_split(
            tensors.numpy(),
            targets,
            test_size=val_split,
            random_state=0,
            shuffle=False,
        )
        self.train_tensors, self.train_targets = train_tensors, train_targets
        return train_tensors, val_tensors, train_targets, val_targets

    def normalize_data(self, train_tensors, val_tensors):
        scaler = StandardScaler()
        scaler.fit(train_tensors)
        train_tensors = torch.tensor(
            scaler.transform(train_tensors), dtype=torch.float32
        )
        val_tensors = torch.tensor(scaler.transform(val_tensors), dtype=torch.float32)
        self.scaler = scaler
        return train_tensors, val_tensors

    def prepare_validation_data(
        self, val_tensors, val_targets, train_tensors, train_targets
    ):
        nearest_y_val = retrieve_nearest_y(train_tensors, train_targets, val_tensors)
        return list(zip(val_tensors, val_targets, nearest_y_val))

    def should_skip_training(self, val_data, loss_weight, verbose):
        current_val_loss = self.compute_val_loss(val_data, loss_weight, verbose)
        if verbose:
            print(f"Initial Validation Loss: {current_val_loss}")
        if (
            hasattr(self, "previous_val_loss")
            and current_val_loss < self.previous_val_loss
        ):
            if verbose:
                print("Validation loss has not degraded. Skipping training.")
            return True
        return False

    def create_dataloader(self, train_tensors, train_targets, batch_size):
        stacked_tensors = train_tensors
        stacked_targets = torch.stack(train_targets).view(
            -1, self.output_sequence_length
        )
        nearest_y_train = retrieve_nearest_y_skip_self(stacked_tensors, stacked_targets)
        stacked_nearest_y = torch.stack(nearest_y_train).view(
            -1, self.output_sequence_length
        )
        dataset = TensorDataset(stacked_tensors, stacked_targets, stacked_nearest_y)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        return dataset, dataloader

    def execute_training(
        self,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        val_data,
        loss_weight,
        patience,
        epochs,
        verbose,
    ):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self.epoch = epoch
            total_loss = 0
            self.mlp.train()
            for batch_x, batch_y, nearest_y in dataloader:
                loss = self.train_single_batch(
                    batch_x, batch_y, nearest_y, criterion, optimizer, loss_weight
                )
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}")

            scheduler.step()
            best_val_loss, patience_counter = self.validate_and_early_stop(
                val_data, best_val_loss, patience_counter, loss_weight, verbose
            )
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping due to no improvement in validation loss.")
                break

        return best_val_loss

    def train_single_batch(
        self, batch_x, batch_y, nearest_y, criterion, optimizer, loss_weight
    ):
        output, features = self.forward(batch_x, batch_y=batch_y, nearest_y=nearest_y)

        mask = torch.ones_like(output)
        masked_output = output * mask

        batch_y = batch_y.view(-1)
        masked_output = masked_output.view(-1, self.num_symbols)

        ce_loss = criterion(masked_output, batch_y)

        if loss_weight > 0:
            contrastive_loss = self.contrastive_loss(features, batch_x)
            loss = ce_loss + loss_weight * contrastive_loss
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def validate_and_early_stop(
        self, val_data, best_val_loss, patience_counter, loss_weight, verbose
    ):
        current_val_loss = self.compute_val_loss(val_data, loss_weight, verbose)
        if verbose:
            print(f"Validation Loss: {current_val_loss}")
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        return best_val_loss, patience_counter

    def compute_val_loss(self, val_data, loss_weight=0.5, verbose=False):
        """
        Compute the validation loss.

        :param val_data: Validation dataset in the form of a list of tuples (tensor, target).
        :param loss_weight: Weight to balance contrastive and ce loss. Set to 0 to ignore contrastive loss.
        :return: Computed validation loss.
        """
        self.mlp.eval()  # Set the model to evaluation mode
        val_tensors, val_targets, nearest_y_val = zip(*val_data)
        val_tensors = torch.stack(val_tensors)
        val_targets = torch.stack(val_targets).view(-1, self.output_sequence_length)
        nearest_y_val = torch.stack(nearest_y_val).view(-1, self.output_sequence_length)

        val_output, val_features = self.forward(val_tensors, nearest_y=nearest_y_val)
        val_mask = torch.ones_like(val_output)
        val_masked_output = val_output * val_mask

        val_targets = val_targets.view(-1)
        val_masked_output = val_masked_output.view(-1, self.num_symbols)

        val_ce_loss = nn.CrossEntropyLoss(ignore_index=self.embedding.padding_idx)(
            val_masked_output, val_targets
        ).item()

        if loss_weight > 0 and self.contrastive_loss_in_val:
            # Calculate contrastive loss for validation
            val_contrastive_loss = self.contrastive_loss(
                val_features, val_tensors
            ).item()

            # Combine the losses for validation loss
            val_loss = val_ce_loss + loss_weight * val_contrastive_loss
        else:
            val_loss = val_ce_loss

        if verbose:
            acc = compute_accuracy(
                val_masked_output, val_targets, self.embedding.padding_idx
            )
            print(f"Validation Loss: {val_loss}, Accuracy: {acc}")
        return val_loss

    def predict(self, semantics, mode="greedy"):
        """
        Predict the output sequence based on the input semantics.

        Parameters:
        - semantics: Input semantics for prediction.
        - mode: Mode of prediction ('greedy' or 'probability').
        """
        number_of_terminals = self.output_sequence_length - self.output_primitive_length
        nearest_y = retrieve_nearest_y(
            self.train_tensors, self.train_targets, semantics.reshape(1, -1)
        )
        nearest_y = torch.stack(nearest_y).view(-1, self.output_sequence_length)

        # Convert semantics to tensor
        semantics_tensor = torch.tensor(semantics, dtype=torch.float32).unsqueeze(
            0
        )  # Add batch dimension

        # Normalize the input semantics using the same scaler as in training
        semantics_tensor = torch.tensor(
            self.scaler.transform(semantics_tensor.numpy()), dtype=torch.float32
        )
        self.mlp.eval()

        # Perform forward pass
        with torch.no_grad():
            output_vector, _ = self.forward(semantics_tensor, nearest_y=nearest_y)

            # Create a mask to ensure that the last four positions are terminals
            mask = torch.full_like(
                output_vector, -float("inf")
            )  # Initialize mask with a large negative value

            # Set the mask values for terminals and non-terminals
            mask[:, :-number_of_terminals] = 0
            mask[
                :, -number_of_terminals:, -self.num_terminals :
            ] = 0  # Allow terminals in last positions
            mask[:, :, self.embedding.padding_idx] = -float(
                "inf"
            )  # Mask out padding index
            mask[:, :, self.start_token_index] = -float("inf")  # Mask out padding index

            # Apply the mask to the output vector
        masked_output_vector = output_vector + mask

        if mode == "probability":
            # Probability sampling mode with resampling
            valid_indices = []
            for i in range(masked_output_vector.size(1)):
                while True:
                    probabilities = torch.softmax(masked_output_vector[:, i, :], dim=1)
                    sampled_index = torch.multinomial(
                        probabilities, num_samples=1
                    ).item()
                    if mask[0, i, sampled_index] != -float("inf"):
                        valid_indices.append(sampled_index)
                        break
            output_indices = np.array(valid_indices)

        elif mode == "greedy":
            # Greedy mode (argmax)
            output_indices = (
                torch.argmax(masked_output_vector, dim=2).squeeze(0).numpy()
            )

        # Get the reverse mapping from indices to node names
        index_to_node_name = self.pset_utils.get_index_to_node_name()

        # Decode indices to node names
        tree_node_names = []
        for i, idx in enumerate(output_indices):
            node_name = index_to_node_name.get(idx, "Unknown")
            tree_node_names.append(node_name)

        return tree_node_names

    def convert_to_primitive_tree(self, semantics, mode="probability"):
        """
        Convert semantics to a GP tree and then to a PrimitiveTree.

        :param semantics: List of node names representing the GP tree.
        :return: PrimitiveTree constructed from the generated GP tree.
        """
        node_names = self.predict(semantics, mode=mode)
        gp_tree = self.pset_utils.convert_node_names_to_gp_tree(node_names)
        return PrimitiveTree(gp_tree)


def generate_synthetic_data(pset, num_samples=100, min_depth=2, max_depth=2):
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
    viewed = set()
    for _ in range(num_samples):
        # Generate a random GP tree
        gp_tree = create_random_gp_tree(pset, min_depth=min_depth, max_depth=max_depth)

        # Generate synthetic semantics (e.g., applying the GP tree to a synthetic dataset)
        target_semantics = normalize_vector(generate_target_semantics(gp_tree, pset))
        if tuple(target_semantics) in viewed:
            continue
        viewed.add(tuple(target_semantics))
        data.append((gp_tree, target_semantics))
        viewed.add(tuple(-1 * target_semantics))
        data.append((gp_tree, -1 * target_semantics))

    # sorting
    # all_semantics = np.argsort(np.median([s for _, s in data], axis=0))
    # for i, (gp_tree, target_semantics) in enumerate(data):
    #     data[i] = (gp_tree, target_semantics[all_semantics])

    return data


def create_random_gp_tree(pset, min_depth=2, max_depth=2):
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


def generate_target_semantics(gp_tree, pset):
    # Compile the GP tree into a callable function
    func = gp.compile(PrimitiveTree(gp_tree), pset)

    # Apply the compiled function to each row of the dataset
    target_semantics = func(*data.T)

    return target_semantics


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def sqrt(x):
    return np.sqrt(np.abs(x))


def aq(x, y):
    return x / np.sqrt(1 + y**2)


def sin(x):
    return np.sin(np.pi * x)


def cos(x):
    return np.cos(np.pi * x)


def multiply(x, y):
    return x * y


def filter_train_data_by_node_count(train_data, max_function_nodes=3):
    """
    Filter out GP trees from train_data with more than a specified number of nodes.

    :param train_data: List of tuples (gp_tree, target_semantics), where gp_tree is a DEAP Individual.
    :param max_function_nodes: Maximum number of nodes allowed in the GP trees.
    :return: List of tuples (gp_tree, target_semantics) with GP trees having nodes count <= max_nodes.
    """
    filtered_train_data = [
        (
            tree,
            semantics / np.linalg.norm(semantics)
            if np.linalg.norm(semantics) > 0
            else semantics,
        )
        for tree, semantics in train_data
        if len([primitive for primitive in tree if isinstance(primitive, Primitive)])
        <= max_function_nodes
    ]
    return filtered_train_data


if __name__ == "__main__":
    # np.random.seed(0)

    # ['subtract', 'add', 'ARG0', 'ARG1', 'subtract', 'ARG0', 'ARG1']
    # Example usage
    data = load_diabetes().data[:50]
    data = StandardScaler().fit_transform(data)

    # Define DEAP PrimitiveSet
    pset = PrimitiveSet(
        "MAIN", data.shape[1]
    )  # Example, replace with your specific pset
    pset.addPrimitive(add, 2)
    pset.addPrimitive(subtract, 2)
    pset.addPrimitive(sin, 1)
    pset.addPrimitive(cos, 1)
    pset.addPrimitive(sqrt, 1)
    pset.addPrimitive(multiply, 2)
    pset.addPrimitive(aq, 2)
    pset.addPrimitive(np.minimum, 2)
    pset.addPrimitive(np.maximum, 2)

    utils = PrimitiveSetUtils(pset)
    generated_tree = PrimitiveTree(
        utils.convert_node_names_to_gp_tree(
            ["subtract", "add", "subtract", "ARG0", "ARG1", "ARG0", "ARG1"]
        )
    )
    utils.convert_gp_tree_to_node_names(generated_tree)
    print([node.name for node in generated_tree])
    print(str(generated_tree))

    # Generate synthetic training data
    fix_depth = 5
    train_data = generate_synthetic_data(
        pset, num_samples=10000, min_depth=1, max_depth=fix_depth
    )

    # Filter training data by node count
    train_data = filter_train_data_by_node_count(train_data, max_function_nodes=3)

    # Split data into training and test sets
    train_data, test_data = train_test_split(train_data, test_size=0.2, random_state=0)

    # Initialize the NeuralSemanticLibrary model
    for use_decoder in [False, True]:
        nl = NeuralSemanticLibrary(
            data.shape[0],
            32,
            32,
            dropout=0.1,
            num_layers=2,
            transformer_layers=1,
            pset=pset,
            use_transformer=True,
            use_decoder_transformer=use_decoder,
            output_primitive_length=3,
            use_x_transformer=False,
        )

        # Train the neural network
        nl.train(
            train_data,
            epochs=1000,
            lr=0.01,
            val_split=0.2,
            verbose=True,
            loss_weight=1,
        )
        for tid in range(0, 5):
            print(
                f"Predicted Tree: {str(nl.convert_to_primitive_tree(test_data[tid][1]))}"
            )
            print(f"Original Tree:  {str(PrimitiveTree(test_data[tid][0]))}")
