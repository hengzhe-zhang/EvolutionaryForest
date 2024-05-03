from typing import List


def construct_important_feature_archive(population: List, external_archive):
    """
    Construct the important feature archive
    :param population: Population
    :param external_archive: External archive
    """
    if external_archive is None:
        external_archive = []
    # Iterate over each tree
    num_of_trees = len(population[0].gene)
    for g in range(num_of_trees):
        if len(external_archive) == num_of_trees:
            # If the external archive is already exist, update the selected features
            selected_features = {str(o[1]): o for o in external_archive[g]}
        else:
            # If the external archive is not exist, create a new dictionary
            selected_features = {}

        # Iterate over individuals in the population
        for p in population:
            # Calculate score for each tree based on fitness and coefficient
            score = p.fitness.wvalues[0] * p.coef[g]
            gene_str = str(p.gene[g])
            if gene_str in selected_features:
                # If the gene is already in the selected features, update the score
                if score > selected_features[gene_str][0]:
                    selected_features[gene_str] = (score, p.gene[g])
                else:
                    pass
            else:
                # If the gene is not in the selected features, add it
                selected_features[gene_str] = (score, p.gene[g])
        # Sort the selected features based on the score
        final_archive = list(
            sorted(selected_features.values(), key=lambda x: x[0], reverse=True)[
                : len(population)
            ]
        )

        if len(external_archive) == num_of_trees:
            external_archive[g] = final_archive
        else:
            external_archive.append(final_archive)
    return external_archive
