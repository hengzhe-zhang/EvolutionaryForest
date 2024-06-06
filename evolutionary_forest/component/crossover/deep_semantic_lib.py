import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from deap import gp, base, creator, tools
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.nn.utils.rnn import pad_sequence

from evolutionary_forest.utility.normalization_tool import normalize_vector


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", ignore_index=-1)

        # Compute pt
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Define GP functions and terminals
def protected_div(a, b):
    return a / b if b != 0 else 1


def setup_deap():
    pset = gp.PrimitiveSet("MAIN", 10)
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genFull, pset=pset, min_=0, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    return toolbox, pset


# Example semantics generator
def get_semantics(individual, inputs, toolbox):
    func = toolbox.compile(expr=individual)
    return [func(*inp) for inp in inputs]


# Neural Network-based Semantic Library
class NeuralSemanticLibrary(nn.Module):
    def __init__(self, input_dim, embed_dim, num_symbols, num_positions):
        super(NeuralSemanticLibrary, self).__init__()
        self.fc1 = nn.Linear(input_dim, embed_dim)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.symbol_embeddings = nn.Parameter(
            torch.randn(num_positions, num_symbols, embed_dim)
        )

    def forward(self, semantics, mask=None):
        semantics_embed = torch.relu(self.bn1(self.fc1(semantics)))
        semantics_embed = torch.relu(self.bn2(self.fc2(semantics_embed)))
        # Ensure semantics_embed has shape (batch_size, num_positions, embed_dim)
        if len(semantics_embed.shape) == 2:
            semantics_embed = semantics_embed.unsqueeze(1).repeat(
                1, self.symbol_embeddings.shape[0], 1
            )
        logits = torch.einsum("bpe,pse->bsp", semantics_embed, self.symbol_embeddings)
        if mask is not None:
            logits = logits.masked_fill(mask.T == 0, -1e9)
        return logits


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    epochs=10,
    patience=10,
    mask=None,
):
    model.train()
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for semantics, target in train_loader:
            optimizer.zero_grad()
            output = model(semantics, mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(output, dim=1)
            valid_indices = target != -1
            total_correct += (
                (predicted[valid_indices] == target[valid_indices]).sum().item()
            )
            total_samples += valid_indices.sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_samples
        avg_val_loss, val_accuracy = evaluate_model(model, criterion, val_loader, mask)

        print(
            f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                model.load_state_dict(torch.load("best_model.pth"))
                break


def evaluate_model(model, criterion, data_loader, mask):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for semantics, target in data_loader:
            output = model(semantics, mask)
            loss = criterion(output, target)
            total_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(output, dim=1)
            valid_indices = target != -1
            total_correct += (
                (predicted[valid_indices] == target[valid_indices]).sum().item()
            )
            total_samples += valid_indices.sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    model.train()
    return avg_loss, accuracy


def prepare_data(toolbox, pset, population, sample_inputs, input_dim, val_split=0.2):
    semantics_list = []
    targets_list = []

    symbol_list = [prim.name for prim in pset.primitives[pset.ret]] + [
        str(term.value) for term in pset.terminals[pset.ret]
    ]

    position_symbol_sets = [set() for _ in range(500)]  # Assuming max length of 50

    target_tracker = {}
    redundant_counter = 0
    for individual in population:
        semantics = get_semantics(individual, sample_inputs, toolbox)
        semantics_tensor = torch.tensor(semantics, dtype=torch.float32)

        target = []
        for idx, node in enumerate(individual):
            symbol = node.name if isinstance(node, gp.Primitive) else str(node.value)
            target.append(symbol_list.index(symbol))
            if idx < 500:
                position_symbol_sets[idx].add(symbol_list.index(symbol))
        target_tensor = torch.tensor(target, dtype=torch.long)

        target_key = tuple(normalize_vector(semantics))
        if target_key in target_tracker:
            redundant_counter += 1
            continue
        else:
            target_tracker[target_key] = True

        semantics_list.append(semantics_tensor)
        targets_list.append(target_tensor)
    print(f"Number of redundant individuals: {redundant_counter}")

    data = pad_sequence(semantics_list, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets_list, batch_first=True, padding_value=-1)
    num_positions = len(targets[0])

    # Create the global mask
    mask = torch.zeros((num_positions, len(symbol_list)), dtype=torch.bool)
    for i, symbol_set in enumerate(position_symbol_sets):
        if len(symbol_set) == 0:
            continue
        mask[i, list(symbol_set)] = True

    # Split into training and validation datasets
    num_val_samples = int(val_split * len(data))
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:-num_val_samples], indices[-num_val_samples:]
    train_data, val_data = data[train_indices], data[val_indices]
    train_targets, val_targets = targets[train_indices], targets[val_indices]

    # Standardize the training data
    scaler = StandardScaler()
    train_data_reshaped = train_data.view(-1, train_data.shape[-1]).numpy()
    train_data_standardized = scaler.fit_transform(train_data_reshaped)
    train_data_standardized = torch.tensor(
        train_data_standardized, dtype=torch.float32
    ).view(train_data.shape)
    val_data_reshaped = val_data.view(-1, val_data.shape[-1]).numpy()
    val_data_standardized = scaler.transform(val_data_reshaped)
    val_data_standardized = torch.tensor(
        val_data_standardized, dtype=torch.float32
    ).view(val_data.shape)

    train_loader = torch.utils.data.DataLoader(
        list(zip(train_data_standardized, train_targets)),
        batch_size=64,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        list(zip(val_data_standardized, val_targets)),
        batch_size=64,
    )

    return train_loader, val_loader, len(symbol_list), num_positions, mask


def main():
    toolbox, pset = setup_deap()
    population = toolbox.population(n=10000)
    input_dim = 10  # Number of inputs in the GP tree
    number_of_data = 50
    embed_dim = 50  # Dimension of the embeddings

    # Generate sample inputs for semantics (replace with your actual input data)
    sample_inputs, _ = load_diabetes(return_X_y=True)
    sample_inputs = sample_inputs[:number_of_data]

    train_loader, val_loader, num_symbols, num_positions, mask = prepare_data(
        toolbox, pset, population, sample_inputs, input_dim
    )

    # decision_tree(train_loader, val_loader)

    model = NeuralSemanticLibrary(number_of_data, embed_dim, num_symbols, num_positions)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding index
    # criterion = FocalLoss()  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = Lion(model.parameters(), lr=1e-3)

    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        epochs=10000,
        mask=mask,
        patience=100,
    )

    # Predict
    model.eval()
    with torch.no_grad():
        # sample_semantics = torch.randn(1, 100)
        first_batch = next(iter(val_loader))
        semantics, target = first_batch
        sample_semantics = semantics[:1]
        target = target[:1]
        print("First batch semantics:", semantics)
        print("First batch target:", target)

        logits = model(sample_semantics, mask)
        probabilities = torch.softmax(logits, dim=1)
        predicted_symbol = torch.argmax(probabilities, dim=1)
        symbol_list = [prim.name for prim in pset.primitives[pset.ret]] + [
            str(term.value) for term in pset.terminals[pset.ret]
        ]
        print(",".join([symbol_list[item] for item in predicted_symbol[0]]))
        print(f"Predicted symbol: {predicted_symbol}")


def decision_tree(train_loader, val_loader):
    # Prepare data for decision tree
    X = []
    y = []
    for semantics, target in train_loader:
        X.append(semantics.numpy())
        y.append(target.numpy())
    for semantics, target in val_loader:
        X.append(semantics.numpy())
        y.append(target.numpy())
    X = np.concatenate(X)
    y_all = np.concatenate(y)
    for dim in range(y_all.shape[1]):
        y = y_all[:, dim]  # Take the first target symbol for simplicity
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        # Train decision tree
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        # Filter out -1 values in ground truth
        valid_indices = y_val != -1
        y_val_filtered = y_val[valid_indices]
        y_pred_filtered = y_pred[valid_indices]

        # Evaluate the decision tree
        accuracy = accuracy_score(y_val_filtered, y_pred_filtered)
        print(f"Decision Tree Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
