import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from deap import gp, base, creator, tools
from lion_pytorch import Lion
from torch.nn.utils.rnn import pad_sequence


# Define GP functions and terminals
def protected_div(a, b):
    return a / b if b != 0 else 1


def setup_deap():
    pset = gp.PrimitiveSet("MAIN", 2)
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addTerminal(1)
    pset.addTerminal(0)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
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
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.symbol_embeddings = nn.Parameter(
            torch.randn(num_positions, num_symbols, embed_dim)
        )

    def forward(self, semantics):
        semantics_embed = torch.relu(self.fc1(semantics))
        semantics_embed = torch.relu(self.fc2(semantics_embed))
        # Ensure semantics_embed has shape (batch_size, num_positions, embed_dim)
        if len(semantics_embed.shape) == 2:
            semantics_embed = semantics_embed.unsqueeze(1).repeat(
                1, self.symbol_embeddings.shape[0], 1
            )
        logits = torch.einsum("bpe,pse->bsp", semantics_embed, self.symbol_embeddings)
        return logits


def train_model(model, criterion, optimizer, data_loader, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for semantics, target in data_loader:
            optimizer.zero_grad()
            output = model(semantics)
            # Flatten output and target to apply cross-entropy loss
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")


def prepare_data(toolbox, pset, population, sample_inputs, input_dim):
    semantics_list = []
    targets_list = []

    symbol_list = [prim.name for prim in pset.primitives[pset.ret]] + [
        str(term.value) for term in pset.terminals[pset.ret]
    ]

    for individual in population:
        semantics = get_semantics(individual, sample_inputs, toolbox)
        semantics_list.append(torch.tensor(semantics, dtype=torch.float32))

        target = []
        for node in individual:
            symbol = node.name if isinstance(node, gp.Primitive) else str(node.value)
            target.append(symbol_list.index(symbol))
        targets_list.append(torch.tensor(target, dtype=torch.long))

    data = pad_sequence(semantics_list, batch_first=True, padding_value=0.0)
    targets = pad_sequence(targets_list, batch_first=True, padding_value=-1)

    dataset = torch.utils.data.TensorDataset(data, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    return data_loader, len(symbol_list), max(len(t) for t in targets_list)


def main():
    toolbox, pset = setup_deap()
    population = toolbox.population(n=50)
    input_dim = 2  # Number of inputs in the GP tree
    number_of_data = 20
    embed_dim = 100  # Dimension of the embeddings

    # Generate sample inputs for semantics (replace with your actual input data)
    sample_inputs = [tuple(np.random.rand(input_dim)) for _ in range(number_of_data)]

    data_loader, num_symbols, num_positions = prepare_data(
        toolbox, pset, population, sample_inputs, input_dim
    )

    model = NeuralSemanticLibrary(number_of_data, embed_dim, num_symbols, num_positions)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding index
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)

    train_model(model, criterion, optimizer, data_loader, epochs=1000)

    # Predict
    model.eval()
    with torch.no_grad():
        # sample_semantics = torch.randn(1, 100)
        first_batch = next(iter(data_loader))
        semantics, target = first_batch
        sample_semantics = semantics[:1]
        target = target[:1]
        print("First batch semantics:", semantics)
        print("First batch target:", target)

        logits = model(sample_semantics)
        probabilities = torch.softmax(logits, dim=1)
        predicted_symbol = torch.argmax(probabilities, dim=1)
        symbol_list = [prim.name for prim in pset.primitives[pset.ret]] + [
            str(term.value) for term in pset.terminals[pset.ret]
        ]
        print(",".join([symbol_list[item] for item in predicted_symbol[0]]))
        print(f"Predicted symbol: {predicted_symbol}")


if __name__ == "__main__":
    main()
