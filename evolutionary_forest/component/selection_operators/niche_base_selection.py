import random

from evolutionary_forest.component.selection import selAutomaticEpsilonLexicaseFast


def niche_base_selection(individuals, k, key_objective=0):
    # Sort individuals based on sharpness (worst to best)
    sorted_individuals = sorted(
        individuals, key=lambda ind: ind.fitness.wvalues[key_objective]
    )

    # Segment the sorted individuals into niches (for simplicity, use sqrt(len(individuals)) niches)
    num_niches = int(len(individuals) ** 0.5)
    niche_size = len(individuals) // num_niches
    niches = [
        sorted_individuals[i : i + niche_size]
        for i in range(0, len(sorted_individuals), niche_size)
    ]

    # Select k individuals using the niching selection
    selected_individuals = []
    while len(selected_individuals) < k:
        niche_index = random.randint(0, num_niches - 1)
        individuals = selAutomaticEpsilonLexicaseFast(niches[niche_index], 2)
        selected_individuals.extend(individuals)

    return selected_individuals[:k]
