import numpy as np


def aggregate_feature_importances(feature_importances, feature_splits):
    if len(feature_importances) != sum(feature_splits):
        raise ValueError(
            f"The sum of feature_splits must match the length of feature_importances. Got {len(feature_importances)} feature importances and {sum(feature_splits)} feature splits."
        )

    aggregated_importances = []
    start_index = 0

    for split in feature_splits:
        # Sum the feature importances for the current GP tree
        aggregated_importances.append(
            sum(feature_importances[start_index : start_index + split])
        )
        # Update the start index for the next GP tree
        start_index += split

    return aggregated_importances


def split_arrays_by_splits(arrays, splits):
    result = []
    start_index = 0

    for split in splits:
        if split == 1:
            result.append(arrays[start_index])
        else:
            result.append(arrays[start_index : start_index + split])
        start_index += split

    return result


def validate_group():
    # Example usage
    arrays = [np.array([1]), np.array([2]), np.array([3]), np.array([4])]
    splits = [1, 1, 2]
    grouped_arrays = split_arrays_by_splits(arrays, splits)
    print("Grouped Arrays:", grouped_arrays)


def validate_feature_aggregation():
    # Example usage
    feature_importances = [0.1, 0.2, 0.3, 0.4]  # Feature importance scores
    feature_splits = [
        1,
        1,
        2,
    ]  # Each GP tree generates 1, 1, and 2 features respectively
    aggregated_importances = aggregate_feature_importances(
        feature_importances, feature_splits
    )
    print("Aggregated Feature Importances:", aggregated_importances)


if __name__ == "__main__":
    validate_feature_aggregation()
    validate_group()
