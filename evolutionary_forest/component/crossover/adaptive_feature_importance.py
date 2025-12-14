import copy
import platform
import random
from functools import wraps
from typing import List, Optional, Tuple

import numpy as np

from evolutionary_forest.multigene_gp import MultipleGeneGP


def normalize_importance(coef: np.ndarray, power: float = 1.0) -> np.ndarray:
    coef_abs = np.abs(coef)
    if coef_abs.sum() == 0:
        return np.ones_like(coef) / len(coef)
    coef_powered = np.power(coef_abs, power)
    return coef_powered / coef_powered.sum()


def revert_genes_by_importance(
    individual: MultipleGeneGP,
    original_genes: List,
    revert_probability: float,
    feature_importance_power: float = 1.0,
):
    """
    Revert genes based on importance.
    """
    if len(original_genes) != len(individual.gene) or individual.coef is None:
        return

    norm_imp = normalize_importance(individual.coef, power=feature_importance_power)

    for i, (orig_gene, imp) in enumerate(zip(original_genes, norm_imp)):
        revert_prob = revert_probability * imp
        if random.random() < revert_prob:
            individual.gene[i] = copy.deepcopy(orig_gene)


def with_revert_probability(
    crossover_configuration,
):
    """
    Decorator for revert probability based on feature importance.
    Supports MAB when revert_probability == "MAB".

    Args:
        crossover_configuration: CrossoverConfiguration object (needed for MAB)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            individuals = [arg for arg in args if isinstance(arg, MultipleGeneGP)]
            if not individuals:
                return func(*args, **kwargs)

            # Use the provided crossover_configuration
            config = crossover_configuration

            # Check config at runtime - if no config, skip
            if config is None:
                return func(*args, **kwargs)

            # Get revert_probability from config
            config_revert_prob = config.revert_probability
            config_feature_power = config.feature_importance_power

            # If no revert_probability, skip
            if not config_revert_prob or config_revert_prob == 0:
                return func(*args, **kwargs)

            # Handle MAB mode
            use_mab = config_revert_prob == "MAB"
            revert_choice_idx = None

            if use_mab:
                # Initialize MAB if not already done
                if (
                    not hasattr(config, "revert_probability_mab")
                    or config.revert_probability_mab is None
                ):
                    config.revert_probability_mab = RevertProbabilityMAB()

                # Select revert_probability using MAB
                actual_revert_prob, revert_choice_idx = (
                    config.revert_probability_mab.select()
                )
            else:
                actual_revert_prob = config_revert_prob

            # Save original genes
            original_genes_list = [copy.deepcopy(ind.gene) for ind in individuals]

            result = func(*args, **kwargs)
            result_individuals = [
                ind
                for ind in (result if isinstance(result, tuple) else [result])
                if isinstance(ind, MultipleGeneGP)
            ]

            if len(result_individuals) != len(original_genes_list):
                return result

            # Store choice_idx for MAB update (if using MAB)
            if use_mab and revert_choice_idx is not None:
                for ind in result_individuals:
                    ind.revert_probability_choice = revert_choice_idx

            # Revert genes based on importance
            for ind, orig_genes in zip(result_individuals, original_genes_list):
                revert_genes_by_importance(
                    ind,
                    orig_genes,
                    actual_revert_prob,
                    config_feature_power,
                )

            return result

        return wrapper

    return decorator


class RevertProbabilityMAB:
    """
    Multi-Armed Bandit for auto-tuning revert probability.
    Tracks parent fitness and choice to determine reward.
    """

    def __init__(
        self, choices: List[float] = [0.0, 1.0, 3.0, 5.0, 7.0], decay_rate: float = 0.99
    ):
        """
        Args:
            choices: List of revert probability values to choose from
            decay_rate: Weight decay rate per generation (0.99 = keep 99%, decay 1%)
        """
        self.choices = choices
        self.decay_rate = decay_rate
        # Track success/failure counts for each choice: [successes, failures]
        self.arm_data = np.ones((2, len(choices)), dtype=float)
        # Track update count to apply decay per generation
        self._update_count = 0
        # Logging on Windows platform
        self._log_enabled = platform.system() == "Windows"
        self._selection_history = []  # Track selections for logging

    def select(self) -> Tuple[float, int]:
        """
        Select a revert probability using Thompson sampling.

        Returns:
            (revert_probability, choice_index)
        """
        # Thompson sampling: sample from beta distribution
        samples = np.random.beta(self.arm_data[0], self.arm_data[1])
        choice_idx: int = int(np.argmax(samples))
        selected_prob = self.choices[choice_idx]

        # Log selection on Windows
        if self._log_enabled:
            self._selection_history.append(choice_idx)

        return (selected_prob, choice_idx)

    def update(
        self,
        parent_fitness: List[float],
        choice_idx: int,
        offspring_fitness: List[float],
    ):
        """
        Update MAB based on reward (fitness improvement).

        Args:
            parent_fitness: Fitness values of parents [fitness1, fitness2]
            choice_idx: Index of the chosen revert probability
            offspring_fitness: Fitness values of offspring [fitness1, fitness2]
        """
        if not parent_fitness or not offspring_fitness:
            return

        # Reward: offspring better than best parent
        best_parent = max(parent_fitness)
        best_offspring = max(offspring_fitness)

        if best_offspring > best_parent:
            self.arm_data[0][choice_idx] += 1  # success
        else:
            self.arm_data[1][choice_idx] += 1  # failure

    def update_from_offspring(self, offspring: List[MultipleGeneGP]):
        """
        Update MAB from a list of offspring.
        Processes offspring in pairs and updates based on fitness improvement.
        Applies weight decay per generation to favor recent rewards.

        Args:
            offspring: List of offspring individuals (should be in pairs from crossover)
        """
        # Apply decay at start of each generation (except first)
        # This gives more weight to recent rewards for dynamic adaptation
        if self._update_count > 0 and self.decay_rate < 1.0:
            self.arm_data *= self.decay_rate
            # Ensure minimum value to avoid zero probabilities
            self.arm_data = np.maximum(self.arm_data, 0.1)

        self._update_count += 1

        # Process in pairs since crossover produces pairs
        for i in range(0, len(offspring) - 1, 2):
            o1, o2 = offspring[i], offspring[i + 1]

            # Check if both have revert_probability_choice (were crossed with MAB)
            if not (
                hasattr(o1, "revert_probability_choice")
                and hasattr(o2, "revert_probability_choice")
            ):
                continue

            choice_idx = o1.revert_probability_choice
            if choice_idx != o2.revert_probability_choice:
                continue  # Should be the same for a pair

            # Get parent fitness (already recorded during crossover)
            if not (hasattr(o1, "parent_fitness") and o1.parent_fitness is not None):
                continue

            # parent_fitness is a tuple: (fitness1, fitness2)
            parent_fitness = list(o1.parent_fitness)

            # Get offspring fitness
            if not (o1.fitness.valid and o2.fitness.valid):
                continue

            offspring_fitness = [o1.fitness.wvalues[0], o2.fitness.wvalues[0]]

            # Update MAB
            self.update(parent_fitness, choice_idx, offspring_fitness)

        # Log statistics on Windows after processing all offspring
        if self._log_enabled and self._update_count > 0:
            self._log_statistics()

    def get_probability(self, choice_idx: Optional[int] = None) -> float:
        """
        Get revert probability for a given choice index, or select one.

        Args:
            choice_idx: Optional choice index, if None selects using MAB

        Returns:
            revert_probability value
        """
        return self.select()[0] if choice_idx is None else self.choices[choice_idx]

    def _log_statistics(self):
        """
        Log MAB statistics on Windows platform.
        Outputs selection rates and success/failure counts.
        """
        if not self._log_enabled:
            return

        # Calculate success rates for each choice
        total_counts = self.arm_data[0] + self.arm_data[1]
        success_rates = np.divide(
            self.arm_data[0],
            total_counts,
            out=np.zeros_like(self.arm_data[0], dtype=float),
            where=total_counts != 0,
        )

        # Count selections in this generation
        if self._selection_history:
            from collections import Counter

            selection_counts = Counter(self._selection_history)
            print(f"\n[MAB] Generation {self._update_count} Statistics:")
            print(f"  Choices: {self.choices}")
            print(f"  Success rates: {[f'{r:.3f}' for r in success_rates]}")
            print(
                f"  Success/Failure counts: {self.arm_data[0].astype(int).tolist()} / {self.arm_data[1].astype(int).tolist()}"
            )
            print(f"  Selections this gen: {dict(selection_counts)}")
            print(f"  Decay rate: {self.decay_rate}")
            self._selection_history.clear()  # Clear for next generation
