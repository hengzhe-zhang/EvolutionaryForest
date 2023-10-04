from evolutionary_forest.strategies.meta_surrogate.individual_level import *

from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone


class GPSurrogate:
    def __init__(self, base_model=None):
        # If no model provided, use DecisionTreeRegressor as default
        self.model = base_model if base_model else DecisionTreeRegressor()
        self.trained = False

    def _compute_features(self, individual):
        """Compute a feature vector for a given GP individual."""
        usage_vector = vectorize_usage(individual, self.all_functions, self.all_terminals)
        return [
            avg_tree_size(individual),
            max_tree_depth(individual),
            avg_function_to_terminal_ratio(individual),
            total_unique_functions(individual),
            total_unique_terminals(individual),
            avg_function_diversity_index(individual),
            avg_terminal_diversity_index(individual)
        ] + usage_vector

    def fit(self, individuals, fitness_values):
        """Train the surrogate model on provided individuals and their fitness values."""
        all_functions, all_terminals = extract_global_vocab(individuals)
        self.all_functions = all_functions  # Store them
        self.all_terminals = all_terminals

        X = [self._compute_features(ind) for ind in individuals]
        y = fitness_values
        self.model = clone(self.model)  # Create a fresh instance of the model
        self.model.fit(X, y)
        self.trained = True

    def predict(self, individuals):
        """Predict the fitness of unseen GP individuals."""
        if not self.trained:
            raise Exception("The surrogate model needs to be trained first.")

        X = [self._compute_features(ind) for ind in individuals]
        return self.model.predict(X)

    def predict_top_k(self, individuals, k):
        """Predict and return the top-k individuals based on their predicted fitness."""
        predictions = self.predict(individuals)

        # Pair each individual with its prediction and sort by prediction
        sorted_individuals = sorted(zip(individuals, predictions), key=lambda x: x[1], reverse=True)

        # Extract the top-k individuals
        top_k_individuals = [ind[0] for ind in sorted_individuals[:k]]

        return top_k_individuals
