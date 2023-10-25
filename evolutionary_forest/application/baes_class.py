import matplotlib.pyplot as plt
import numpy as np  # noqa
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def spearman(ya, yb):
    correlation = spearmanr(ya, yb)[0]
    return correlation


class ExpensiveMOEABenchmark:
    def __init__(self):
        self.first_layer_funcs = []
        self.second_layer_funcs = []

    def generate_data(self):
        X0 = np.random.rand(50, 10)
        vars = [
            np.mean(np.apply_along_axis(func, 1, X0), axis=1)
            for func in self.first_layer_funcs
        ]
        VAR0 = vars[0]
        VAR1 = vars[1]
        VAR2 = vars[2]
        VAR3 = vars[3]
        VAR4 = vars[4]
        O1, O2 = [
            func(VAR0, VAR1, VAR2, VAR3, VAR4) for func in self.second_layer_funcs
        ]
        return X0, O1, O2

    def plot(self):
        # Generate the x, y, and z values for the plot
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        for function in self.second_layer_funcs:
            vars = [
                np.mean(
                    np.apply_along_axis(
                        func, 1, np.stack([X.flatten(), Y.flatten()]).T
                    ),
                    axis=1,
                )
                for func in self.first_layer_funcs
            ]
            Z = function(*vars)
            Z = Z.reshape(X.shape[0], X.shape[1])

            # Create a 3D plot of the function
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()

    def evaluation(self):
        X0, O1, O2 = self.generate_data()
        pipe = Pipeline(
            [
                ("Scaler", StandardScaler()),
                ("GP", GaussianProcessRegressor(normalize_y=True)),
            ]
        )
        print(
            np.mean(cross_val_score(pipe, X0, O1, scoring=make_scorer(spearman))),
            np.mean(cross_val_score(pipe, X0, O2, scoring=make_scorer(spearman))),
        )
