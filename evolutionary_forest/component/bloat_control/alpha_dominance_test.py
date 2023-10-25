import math

from evolutionary_forest.component.bloat_control.alpha_dominance import AlphaDominance


# Mocking EvolutionaryForestRegressor
class EvolutionaryForestRegressor:
    def __init__(self):
        # Assume it has a dictionary named 'bloat_control' with a key 'step_size'
        self.bloat_control = {"step_size": 0.1}

    hof = []


# Mocking a simple individual class
class Individual:
    def __init__(self, size, fitness_value):
        self._size = size
        self.fitness = self.Fitness(fitness_value)

    class Fitness:
        def __init__(self, value):
            self.values = (value,)
            self.weights = (-1,)
            self.wvalues = tuple(
                val * weight for val, weight in zip(self.values, self.weights)
            )

        def dominates(self, other, obj=None):
            if obj is None:
                obj = self.values
            return all(a <= b for a, b in zip(obj, other.values)) and any(
                a < b for a, b in zip(obj, other.values)
            )

    def __len__(self):
        return self._size


# Create a population and offspring
population = [Individual(5, 0.8), Individual(7, 0.9)]
offspring = [Individual(6, 0.7), Individual(8, 0.85)]

# Create an instance of AlphaDominance
algorithm = EvolutionaryForestRegressor()
dominance = AlphaDominance(algorithm)

# Update best historical values
dominance.update_best(population)

# Run selection and get the new alpha value
alpha = 0.5
new_alpha = dominance.selection(population, offspring, alpha)
print(f"New Alpha: {new_alpha}")
