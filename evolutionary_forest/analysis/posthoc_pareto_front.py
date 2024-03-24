import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from evolutionary_forest.component.decision_making.bend_angle_knee import (
    find_knee_based_on_bend_angle,
)
from evolutionary_forest.component.environmental_selection import EnvironmentalSelection
from evolutionary_forest.component.fitness import RademacherComplexityR2, R2PACBayesian
from evolutionary_forest.multigene_gp import *
from evolutionary_forest.utility.sliced_predictor import SlicedPredictor
from evolutionary_forest.utils import pareto_front_2d

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class ParetoFrontTool:
    @staticmethod
    def calculate_test_pareto_front(
        self: "EvolutionaryForestRegressor", test_x, test_y
    ):
        """
        Calculate the Pareto front for test data based on the evolutionary forest regressor.

        This function determines the Pareto front from the current population of the model and
        computes the normalized prediction error for the test dataset. The results are appended
        to the `pareto_front` attribute of the `EvolutionaryForestRegressor` object.

        Parameters:
        - self (EvolutionaryForestRegressor): An instance of the EvolutionaryForestRegressor class.
        - test_x (array-like): Test input data. Each row represents an instance and each column represents a feature.
        - test_y (array-like): True target values for the test input data.
        """
        # Calculate normalization factors for scaling and test dataset
        normalization_factor_scaled = np.mean((self.y - np.mean(self.y)) ** 2)
        normalization_factor_test = np.mean((test_y - np.mean(test_y)) ** 2)

        multiobjective_plot = False
        if multiobjective_plot:
            fitness_values = [ind.fitness.wvalues for ind in self.pop]
            # Extract x and y coordinates from fitness values
            x = np.array([fitness[0] for fitness in fitness_values])
            y = np.array([fitness[1] for fitness in fitness_values])
            knee_index = np.argmax(x + y)

            # Create scatter plot
            plt.scatter(x / (x.max() - x.min()), y / (y.max() - y.min()))
            plt.scatter(
                x[knee_index] / (x.max() - x.min()),
                y[knee_index] / (y.max() - y.min()),
                color="red",
                label="Knee Point",
            )
            plt.title("Fitness Scatter Plot")
            plt.xlabel("Fitness Value 1")
            plt.ylabel("Fitness Value 2")
            plt.grid(True)
            plt.show()

        # Compute normalized prediction error for each individual
        for ind in self.pop:
            prediction = self.individual_prediction(test_x, [ind])[0]
            errors = (test_y - prediction) ** 2
            test_error_normalized_by_test = np.mean(errors) / normalization_factor_test
            self.pareto_front.append(
                (
                    float(np.mean(ind.case_values) / normalization_factor_scaled),
                    float(test_error_normalized_by_test),
                )
            )

            # feature construction
            train_x = self.X
            constructed_train_x = self.feature_generation(train_x, ind)
            # scaling test data
            scaled_test_x = self.x_scaler.transform(test_x)
            constructed_test_x = self.feature_generation(scaled_test_x, ind)

            test_error_normalized_knn = ParetoFrontTool.model_transfer(
                self,
                normalization_factor_test,
                constructed_train_x,
                constructed_test_x,
                test_y,
                model_name="KNN",
            )
            test_error_normalized_wknn = ParetoFrontTool.model_transfer(
                self,
                normalization_factor_test,
                constructed_train_x,
                constructed_test_x,
                test_y,
                model_name="WKNN",
            )
            test_error_normalized_dt = ParetoFrontTool.model_transfer(
                self,
                normalization_factor_test,
                constructed_train_x,
                constructed_test_x,
                test_y,
                model_name="DT",
            )
            self.knn_pareto_front.append(
                (
                    float(test_error_normalized_by_test),
                    float(test_error_normalized_knn),
                )
            )
            self.wknn_pareto_front.append(
                (
                    float(test_error_normalized_by_test),
                    float(test_error_normalized_wknn),
                )
            )
            self.dt_pareto_front.append(
                (
                    float(test_error_normalized_by_test),
                    float(test_error_normalized_dt),
                )
            )
            del prediction
            del errors
        self.pareto_front, _ = pareto_front_2d(self.pareto_front)
        self.pareto_front = self.pareto_front.tolist()

        self.knn_pareto_front, _ = pareto_front_2d(self.knn_pareto_front)
        self.knn_pareto_front = self.knn_pareto_front.tolist()

        self.wknn_pareto_front, _ = pareto_front_2d(self.wknn_pareto_front)
        self.wknn_pareto_front = self.wknn_pareto_front.tolist()

        self.dt_pareto_front, _ = pareto_front_2d(self.dt_pareto_front)
        self.dt_pareto_front = self.dt_pareto_front.tolist()

    @staticmethod
    def model_transfer(
        self,
        normalization_factor_test,
        constructed_train_x,
        constructed_test_x,
        test_y,
        model_name="KNN",
    ):
        train_y = self.y
        if model_name == "KNN":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SlicedPredictor(KNeighborsRegressor(n_neighbors=5))),
                ]
            )
        elif model_name == "WKNN":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        SlicedPredictor(
                            KNeighborsRegressor(n_neighbors=5, weights="distance")
                        ),
                    ),
                ]
            )
        elif model_name == "DT":
            model = SlicedPredictor(DecisionTreeRegressor())
        else:
            raise Exception("Model name is not supported")
        model.fit(constructed_train_x, train_y)

        prediction = model.predict(constructed_test_x)
        prediction = self.y_scaler.inverse_transform(
            prediction.reshape(-1, 1)
        ).flatten()
        errors = (test_y - prediction) ** 2
        test_error_normalized_transfer_model = (
            np.mean(errors) / normalization_factor_test
        )
        return test_error_normalized_transfer_model

    @staticmethod
    def calculate_sharpness_pareto_front(self: "EvolutionaryForestRegressor", x_train):
        if self.experimental_configuration.pac_bayesian_comparison and isinstance(
            self.environmental_selection, EnvironmentalSelection
        ):
            pac = R2PACBayesian(self, **self.param)

            # Calculate normalization factors
            normalization_factor_scaled = np.mean((self.y - np.mean(self.y)) ** 2)

            # Compute sharpness and other metrics for each individual
            for ind in self.pop:
                ParetoFrontTool.sharpness_estimation(self, ind, pac)
                sharpness_value = ind.fitness_list[1][0]
                """
                1. Normalized fitness values (case values represent cross-validation squared error)
                2. Normalized sharpness values
                """
                self.pareto_front.append(
                    (
                        float(np.mean(ind.case_values) / normalization_factor_scaled),
                        float(sharpness_value / normalization_factor_scaled),
                    )
                )

            self.pareto_front, _ = pareto_front_2d(self.pareto_front)

    @staticmethod
    def sharpness_estimation(self, ind, pac, force=False):
        if not isinstance(self.score_func, R2PACBayesian) or force:
            if isinstance(self.score_func, RademacherComplexityR2):
                ind.rademacher_fitness_list = ind.fitness_list
            pac.assign_complexity(ind, ind.pipe)

    @staticmethod
    def calculate_sharpness_pareto_front_on_test(
        self: "EvolutionaryForestRegressor", test_x, test_y
    ):
        if self.experimental_configuration.pac_bayesian_comparison and isinstance(
            self.environmental_selection, EnvironmentalSelection
        ):
            pac = R2PACBayesian(self, **self.param)
            if isinstance(self, ClassifierMixin):
                pac.classification = True
                pac.instance_weights = self.class_weight
            self.training_test_pareto_front = []
            self.test_pareto_front = []
            """
            very important:
            Please ensure training data and test data are scaled in the same way.
            If test data is calculated on unscaled data, whereas training data is calculated on scaled data,
            the result will be wrong.
            """
            # Calculate normalization factors
            normalization_factor_test = np.mean((test_y - np.mean(test_y)) ** 2)
            normalization_factor_scaled = np.mean((self.y - np.mean(self.y)) ** 2)
            # Get scaled test data
            if self.x_scaler is not None:
                test_X_scaled = self.x_scaler.transform(test_x)
                test_y_scaled = self.y_scaler.transform(test_y.reshape(-1, 1)).flatten()
            else:
                test_X_scaled = test_x
                test_y_scaled = test_y
            normalization_factor_test_scaled = np.mean(
                (test_y_scaled - np.mean(test_y_scaled)) ** 2
            )

            # Compute sharpness and other metrics for each individual
            for ind in self.pop:
                test_sharpness_estimation = False
                if test_sharpness_estimation:
                    # calculate sharpness based on test data
                    # backup
                    back_up_x = self.X
                    back_up_y = self.y
                    backup_fitness_list = ind.fitness_list
                    # estimate sharpness
                    self.X = test_X_scaled
                    self.y = test_y_scaled
                    # the sharpness is calculated based on scaled y
                    ParetoFrontTool.sharpness_estimation(self, ind, pac, force=True)
                    # the sharpness across all samples on test data
                    sharpness_value = ind.fitness_list[1][0]
                    # restore
                    ind.fitness_list = backup_fitness_list
                    self.X = back_up_x
                    self.y = back_up_y
                else:
                    # calculate sharpness based on training data
                    ParetoFrontTool.sharpness_estimation(self, ind, pac)
                    sharpness_value = ind.fitness_list[1][0]

                prediction = self.individual_prediction(test_x, [ind])[0]

                errors = (test_y - prediction) ** 2
                assert len(errors) == len(test_y)
                """
                1. Normalized test error (1-Test R2)
                2. Normalized sharpness values / model size
                3. Normalized test error, but use same normalization factor as training data
                """
                test_error_normalized_by_test = (
                    np.mean(errors) / normalization_factor_test
                )
                if test_sharpness_estimation:
                    # sharpness should be scaled on scaled y because it is calculated on scaled y
                    sharpness_normalized = float(
                        sharpness_value / normalization_factor_test_scaled
                    )
                else:
                    # if sharpness is calculated on training data, it should be scaled on training data
                    sharpness_normalized = float(
                        sharpness_value / normalization_factor_scaled
                    )
                self.test_pareto_front.append(
                    (
                        test_error_normalized_by_test,
                        sharpness_normalized,
                    )
                )
                self.size_pareto_front.append(
                    (
                        test_error_normalized_by_test,
                        sum([len(gene) for gene in ind.gene]),
                    )
                )
                training_fitness_normalized = float(
                    np.mean(ind.case_values) / normalization_factor_scaled
                )
                self.training_test_pareto_front.append(
                    (
                        training_fitness_normalized,
                        sum([len(gene) for gene in ind.gene]),
                        test_error_normalized_by_test,
                    )
                )
                del prediction
                del errors

            self.test_pareto_front, _ = pareto_front_2d(self.test_pareto_front)
            self.size_pareto_front, _ = pareto_front_2d(self.size_pareto_front)
            self.test_pareto_front = self.test_pareto_front.tolist()
            self.size_pareto_front = self.size_pareto_front.tolist()

            # the information of Pareto front
            self.training_test_pareto_front = np.array(self.training_test_pareto_front)
            _, indices = pareto_front_2d(self.training_test_pareto_front[:, :2])
            self.training_test_pareto_front = self.training_test_pareto_front[indices]
            self.training_test_pareto_front = self.training_test_pareto_front.tolist()


def create_scatter_plot(data, color_map="viridis"):
    data[:, :2] = (data[:, :2] - data[:, :2].min(axis=0)) / (
        data[:, :2].max(axis=0) - data[:, :2].min(axis=0)
    )
    _, traditional_knee = find_knee_based_on_bend_angle(data[:, :2], local=True)
    _, complexity_knee, knee_index = find_knee_based_on_bend_angle(
        data[:, :2],
        local=True,
        number_of_cluster=3,
        return_all_knees=True,
        minimal_complexity=True,
    )

    # Extract x, y, and color values
    x = [point[0] for point in data]
    y = [point[1] for point in data]
    # Exchange X and Y
    x, y = y, x
    colors = [point[2] for point in data]

    # Create colormap object and normalize colors
    colormap = cm.get_cmap(color_map)
    normalize = Normalize(vmin=min(colors), vmax=max(colors))

    # Highlight points on the scatter plot
    sns.scatterplot(x=x, y=y, hue=colors, palette=color_map)
    legend = plt.legend(title="Test RSE")

    # Convert knee_index to integers for indexing
    knee_index_int = [int(k) for k in knee_index]

    # Annotate knee points with letters
    for i, index in enumerate(knee_index_int):
        plt.annotate(
            chr(ord("A") + i),
            (x[index], y[index]),
            fontsize=12,
            color="red",
            weight="bold",
        )

    # Get the color for traditional_knee and complexity_knee from the color map
    traditional_knee_color = colormap(normalize(colors[traditional_knee]))
    complexity_knee_color = colormap(normalize(colors[complexity_knee]))

    # Highlight traditional_knee with a special marker (e.g., a star)
    plt.scatter(
        x[traditional_knee],
        y[traditional_knee],
        marker="*",
        s=100,
        c=traditional_knee_color,
        label="Traditional Knee",
    )

    # Highlight complexity_knee with a different special marker (e.g., a diamond)
    plt.scatter(
        x[complexity_knee],
        y[complexity_knee],
        marker="D",
        s=50,
        c=complexity_knee_color,
        label="Complexity Knee",
    )

    plt.xlabel("Objective B: Complexity")
    plt.ylabel("Objective A: Training Error")
