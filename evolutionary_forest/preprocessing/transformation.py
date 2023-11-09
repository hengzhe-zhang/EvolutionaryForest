import numpy as np
import openml
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder


class MeanMedianFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, numerical_columns):
        self.categorical_features = categorical_features
        self.numerical_columns = numerical_columns
        self.aggregations_ = None  # to store training data aggregations
        self.categorical_indicator_ = []  # to store which new features are categorical

    def fit(self, X, y=None):
        self.categorical_indicator_.clear()
        if len(self.numerical_columns) == 0:
            return self

        # Compute the aggregations on the training data
        self.aggregations_ = {}
        for feature in self.categorical_features:
            aggregated = X.groupby(feature)[self.numerical_columns].agg(
                ["mean", "median", "std", "max", "min"]
            )
            # Flatten the MultiIndex in columns
            aggregated.columns = [
                f"{feature}_{func}_{col}" for col, func in aggregated.columns
            ]
            self.aggregations_[feature] = aggregated
            # Since the new features are numerical, extend the indicator with False
            self.categorical_indicator_.extend([False] * len(aggregated.columns))
        return self

    def transform(self, X):
        if len(self.numerical_columns) == 0:
            return X

        # Apply the aggregations to the test data
        X_merged = X.copy()
        for feature in self.categorical_features:
            aggregated = self.aggregations_[feature]
            X_merged = X_merged.merge(
                aggregated, left_on=feature, right_index=True, how="left"
            )
        X_merged.drop(columns=X.columns, inplace=True)
        return X_merged

    def get_categorical_indicator(self):
        # Return the indicator of which features are categorical
        return self.categorical_indicator_


# Define a custom function to get the mode
def get_mode(s):
    mode_val = s.mode()
    if mode_val.empty:
        return None
    else:
        return mode_val.iloc[0]


class ModeFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features):
        self.categorical_features = categorical_features
        self.aggregations_ = None
        self.categorical_indicator_ = []

    def fit(self, X, y=None):
        self.categorical_indicator_.clear()
        self.aggregations_ = {}
        if len(self.categorical_features) > 1:
            for feature in self.categorical_features:
                # Exclude the current feature from aggregation
                features_to_agg = [f for f in self.categorical_features if f != feature]
                # Calculate mode, nunique, and count
                aggregated = X.groupby(feature)[features_to_agg].agg(
                    [get_mode, "nunique", "count"]
                )
                # Flatten the MultiIndex and create new column names
                aggregated.columns = [
                    f"{feature}_{agg_func}_{col}"
                    for col in features_to_agg
                    for agg_func in ["mode", "nunique", "count"]
                ]
                self.aggregations_[feature] = aggregated

                # Initialize the categorical indicator
                self.categorical_indicator_.extend(
                    [True, False, False] * len(features_to_agg),
                )
        return self

    def transform(self, X):
        X_merged = X.copy()
        if len(self.categorical_features) > 1:
            for feature in self.categorical_features:
                # Ensure the feature is in the dataset
                if feature in X:
                    aggregated = self.aggregations_[feature]
                    # Perform a left join with the original DataFrame
                    X_merged = X_merged.merge(
                        aggregated, left_on=feature, right_index=True, how="left"
                    )
        X_merged.drop(columns=X.columns, inplace=True)
        return X_merged

    def get_categorical_indicator(self):
        # Return the indicator of which features are categorical
        return self.categorical_indicator_


class CombineFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, max_unique_values=100):
        self.categorical_features = categorical_features
        self.max_unique_values = max_unique_values
        self.combination_mappings = {}
        self.combination_value_counts = {}
        self.categorical_indicator_ = []

    def fit(self, X, y=None):
        self.combination_mappings.clear()
        self.combination_value_counts.clear()
        self.categorical_indicator_.clear()
        # Find all valid combinations of features in the training set
        for i, feature_a in enumerate(self.categorical_features):
            for feature_b in self.categorical_features[i + 1 :]:
                combined_series = (
                    X[feature_a].astype(str) + "_" + X[feature_b].astype(str)
                )
                if combined_series.nunique() <= self.max_unique_values:
                    # Factorize the combined series and store the mapping
                    labels, unique = pd.factorize(combined_series)
                    mapping = dict(zip(unique, labels))
                    self.combination_mappings[(feature_a, feature_b)] = mapping
                    # Store value counts for the combined labels
                    label_counts = pd.Series(labels).value_counts()
                    self.combination_value_counts[(feature_a, feature_b)] = label_counts
                    # Indicate that the new feature is categorical/numerical
                    self.categorical_indicator_.extend([True, False])
        return self

    def transform(self, X):
        X_merged = X.copy()
        for (feature_a, feature_b), mapping in self.combination_mappings.items():
            # Combine the two Series into a new Series
            combined = X[feature_a].astype(str) + "_" + X[feature_b].astype(str)
            # Map the combined series based on the factorized labels from fit
            X_merged[f"{feature_a}+{feature_b}"] = (
                combined.map(mapping).fillna(-1).astype(int)
            )
            # Use the pre-calculated value counts from fit
            value_counts = self.combination_value_counts[(feature_a, feature_b)]
            X_merged[f"{feature_a}+{feature_b}/value_count"] = (
                X_merged[f"{feature_a}+{feature_b}"].map(value_counts).fillna(0)
            )
        X_merged.drop(columns=X.columns, inplace=True)
        return X_merged

    def get_categorical_indicator(self):
        # Return the indicator of which features are categorical
        return self.categorical_indicator_


class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_indicator):
        self.categorical_indicator = categorical_indicator
        self.encoder = None

    def fit(self, X, y=None):
        # Identify categorical columns based on categorical_indicator
        self.categorical_columns = [
            X.columns[i]
            for i, is_categorical in enumerate(self.categorical_indicator)
            if is_categorical
        ]
        # Initialize OneHotEncoder from sklearn and fit it on the categorical columns
        self.encoder = OneHotEncoder(
            drop="first", sparse=False, handle_unknown="ignore"
        )
        self.encoder.fit(X[self.categorical_columns])
        return self

    def transform(self, X):
        # Check if the encoder has been fitted
        if self.encoder is None:
            raise RuntimeError("You must fit the transformer before calling transform")

        # One-hot encode the categorical columns
        X_encoded = self.encoder.transform(X[self.categorical_columns])

        # Convert to DataFrame
        X_encoded_df = pd.DataFrame(
            X_encoded,
            index=X.index,
            columns=self.encoder.get_feature_names_out(),
        )

        # Drop original categorical columns and concatenate the new one-hot encoded columns
        X_non_categorical = X.drop(columns=self.categorical_columns)
        X_merged = pd.concat([X_non_categorical, X_encoded_df], axis=1)

        return X_merged


class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_indicator):
        self.categorical_indicator_ = categorical_indicator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def get_categorical_indicator(self):
        return self.categorical_indicator_


class DataFrameFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        result = super().fit_transform(X, y=y, **fit_params)
        # Get feature names from all transformers
        columns = self.get_feature_names_out(X=X)
        return pd.DataFrame(result, index=X.index, columns=columns)

    def transform(self, X):
        result = super().transform(X)
        # Get feature names from all transformers
        columns = self.get_feature_names_out(X=X)
        return pd.DataFrame(result, index=X.index, columns=columns)

    def get_feature_names_out(self, **kwargs):
        X = kwargs["X"]
        feature_names = []
        for name, trans in self.transformer_list:
            if hasattr(trans, "get_feature_names_out"):
                trans_feature_names = trans.get_feature_names_out()
            else:
                # If no method 'get_feature_names_out', generate names sequentially
                trans_feature_names = [
                    f"{name}__x{idx}" for idx in range(trans.fit_transform(X).shape[1])
                ]
            feature_names.extend(trans_feature_names)
        return feature_names


def get_processing_transformer(
    categorical_indicator,
    categorical_features,
    numerical_columns,
    use_mean_features=False,
    use_mode_features=False,
    use_combine_features=False,
) -> FeatureUnion:
    # Initialize transformers
    mean_median_transformer = MeanMedianFeaturesTransformer(
        categorical_features=categorical_features, numerical_columns=numerical_columns
    )
    mode_features_transformer = ModeFeaturesTransformer(
        categorical_features=categorical_features
    )
    combine_features_transformer = CombineFeaturesTransformer(
        categorical_features=categorical_features
    )
    # Create a FeatureUnion of transformers
    transformer_list = [
        ("identity", IdentityTransformer(categorical_indicator)),
    ]
    if use_mean_features:
        transformer_list.append(
            ("mean_median", mean_median_transformer),
        )
    if use_mode_features:
        transformer_list.append(
            ("mode_features", mode_features_transformer),
        )
    if use_combine_features:
        transformer_list.append(
            ("combine_features", combine_features_transformer),
        )
    combined_transformer = DataFrameFeatureUnion(transformer_list=transformer_list)
    return combined_transformer


def load_data(dataset_id):
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    x, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="dataframe"
    )
    categorical_indicator_np = np.array(categorical_indicator)
    categorical_features = x.columns[categorical_indicator_np]
    numerical_columns = x.columns[~categorical_indicator_np]
    return x, y, categorical_indicator, categorical_features, numerical_columns


def fit_and_transform(
    X_train,
    X_test,
    categorical_indicator,
    categorical_features,
    numerical_columns,
    use_mean_features=False,
    use_mode_features=False,
    use_combine_features=False,
):
    combined_transformer = get_processing_transformer(
        categorical_indicator,
        categorical_features,
        numerical_columns,
        use_mean_features=use_mean_features,
        use_mode_features=use_mode_features,
        use_combine_features=use_combine_features,
    )
    combined_transformer.fit(X_train)  # Fit the transformer to the training data
    X_train_transformed = combined_transformer.transform(
        X_train
    )  # Transform training data
    X_test_transformed = combined_transformer.transform(X_test)  # Transform test data

    categorical_indicators = []
    for name, transformer in combined_transformer.transformer_list:
        categorical_indicators.extend(transformer.get_categorical_indicator())
    return X_train_transformed, X_test_transformed, categorical_indicators


if __name__ == "__main__":
    dataset_id = 44987
    x, y, categorical_indicator, categorical_features, numerical_columns = load_data(
        dataset_id=44987
    )

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    X_train_transformed, X_test_transformed, categorical_indicators = fit_and_transform(
        X_train,
        X_test,
        categorical_indicator,
        categorical_features,
        numerical_columns,
    )

    print(X_train_transformed)
    print(X_test_transformed)
    print(categorical_indicators)
    assert (
        len(categorical_indicators) == X_train_transformed.shape[1]
    ), f"{len(categorical_indicators)},{X_train_transformed.shape[1]}"
    one_hot = OneHotEncoderCustom(categorical_indicator)
    one_hot.fit(X_train_transformed)
    print(one_hot.transform(X_train_transformed))
