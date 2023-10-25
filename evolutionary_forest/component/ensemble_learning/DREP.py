import numpy as np


class DREPEnsemble:
    def __init__(self, y_data, predictions, algorithm):
        self.y_sample = y_data.flatten()
        self.predictions = predictions
        self.algorithm = algorithm
        self.ensemble_weights = np.zeros(len(self.algorithm.hof))

    def generate_ensemble_weights(self):
        # Initialize a prediction vector with zeros (same shape as y_sample)
        current_prediction = np.zeros_like(self.y_sample)

        # Create a set of indices representing models in the algorithm's hall of fame
        remain_ind = set(range(len(self.algorithm.hof)))
        min_index = 0

        # Find the base model with the smallest error with respect to y_sample
        for i in remain_ind:
            error = np.mean((self.predictions[i] - self.y_sample) ** 2)
            if error < np.mean((current_prediction - self.y_sample) ** 2):
                current_prediction = self.predictions[i]
                min_index = i
        # Remove the index of the selected base model from the remaining set
        remain_ind.remove(min_index)

        # Set the weight of the selected base model to 1 in the ensemble weights
        self.ensemble_weights[min_index] = 1

        # Continue the process of adding models to the ensemble until no further improvement is observed
        while True:
            div_list = []

            # Calculate the diversity (difference) and loss (error) of each model relative to the current ensemble prediction
            for i in remain_ind:
                diversity = np.mean(((current_prediction - self.predictions[i]) ** 2))
                loss = np.mean(((self.y_sample - self.predictions[i]) ** 2))
                div_list.append((diversity, loss, i))

            # Filter the models by diversity, retaining the top 50% diverse models
            div_list = sorted(div_list, key=lambda x: -x[0])[
                : int(round(len(div_list) * 0.5))
            ]

            # Sort the shortlisted models by their loss (ascending)
            div_list = sorted(div_list, key=lambda x: x[1])

            # If no models remain after filtering and sorting, exit the loop
            if not div_list:
                break

            # Extract the index of the model with the smallest loss among the shortlisted models
            index = div_list[0][2]

            # Calculate the new ensemble prediction if the model were to be added
            ensemble_size = np.sum(self.ensemble_weights)
            trial_prediction = (
                ensemble_size / (ensemble_size + 1) * current_prediction
                + 1 / (ensemble_size + 1) * self.predictions[index]
            )

            # If the new ensemble prediction doesn't improve performance, exit the loop
            if np.mean(((trial_prediction - self.y_sample) ** 2)) > np.mean(
                ((current_prediction - self.y_sample) ** 2)
            ):
                break

            # Update the ensemble prediction and set the weight of the selected model to 1
            current_prediction = trial_prediction
            self.ensemble_weights[index] = 1
            remain_ind.remove(index)

        # Normalize the ensemble weights so that they sum up to 1
        self.ensemble_weights /= np.sum(self.ensemble_weights)
        # Update the tree_weight attribute of the algorithm with the computed ensemble weights
        self.algorithm.tree_weight = self.ensemble_weights

    def get_ensemble_weights(self):
        return self.ensemble_weights
