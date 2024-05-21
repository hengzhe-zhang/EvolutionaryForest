import numpy as np

from evolutionary_forest.utility.statistics.variance_tool import mean_without_outliers


def collect_information_of_sharpness(all_individuals):
    all_mse = []
    all_sharpness = []
    for ind in all_individuals:
        if hasattr(ind, "fitness_list"):
            # minimize
            sharpness = ind.fitness_list[1][0]
        else:
            sharpness = -1 * ind.fitness.wvalues[1]
        naive_mse = np.mean(ind.case_values)
        all_mse.append(naive_mse)
        all_sharpness.append(sharpness)
    return all_mse, all_sharpness


def weighted_sam_loss(all_individuals, ratio):
    for ind in all_individuals:
        if hasattr(ind, "fitness_list"):
            # minimize
            sharpness = ind.fitness_list[1][0]
        else:
            sharpness = -1 * ind.fitness.wvalues[1]
        # naive_mse = np.mean(ind.case_values)
        naive_mse = ind.naive_mse
        ind.sam_loss = naive_mse + ratio * sharpness


def balanced_ratio(all_mse, all_sharpness, metric_std):
    mean_of_error = mean_without_outliers(np.array(all_mse), metric=metric_std)
    mean_of_sam = mean_without_outliers(np.array(all_sharpness), metric=metric_std)
    ratio = mean_of_error / mean_of_sam
    return mean_of_error, mean_of_sam, ratio
