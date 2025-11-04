from evolutionary_forest.model.OptimalKNN import OptimalKNN
from evolutionary_forest.model.knn.FaissKNNRegressor import (
    FaissKNNLinearRankRegressor,
    FaissKNNRankRegressor,
    RobustFaissKNNRegressor,
    AdaptiveRNRegressor,
)
from evolutionary_forest.model.optimal_knn.GBOptimalKNN import RidgeBoostedKNN


def adaptive_neighbors(base_learner, params, X, y):
    if (
        base_learner in ["RidgeBoosted-LinearRankKNN", "OptimalLinearRankKNN"]
        and params["n_neighbors"] == "Adaptive"
    ):
        if X.shape[0] <= 200:
            params["n_neighbors"] = 5
        else:
            params["n_neighbors"] = 30


def get_knn_model(base_learner, params):
    if base_learner == "RidgeBoosted-LinearRankKNN":
        ridge_model = RidgeBoostedKNN(
            knn_params={
                **params,
                "base_learner": (
                    lambda: FaissKNNLinearRankRegressor(
                        n_neighbors=params["n_neighbors"]
                    )
                )(),
            }
        )
        return ridge_model
    elif base_learner == "RidgeBoosted-RankKNN":
        ridge_model = RidgeBoostedKNN(
            knn_params={
                **params,
                "base_learner": (
                    lambda: FaissKNNRankRegressor(n_neighbors=params["n_neighbors"])
                )(),
            }
        )
        return ridge_model
    elif base_learner == "RidgeBoosted-RN":
        ridge_model = RidgeBoostedKNN(
            knn_params={
                **params,
                "base_learner": (
                    lambda: AdaptiveRNRegressor(radius=params["n_neighbors"])
                )(),
            }
        )
        return ridge_model
    elif base_learner == "OptimalLinearRankKNN":
        ridge_model = OptimalKNN(
            base_learner=FaissKNNLinearRankRegressor(n_neighbors=params["n_neighbors"])
        )
        return ridge_model
    raise ValueError(f"Unknown base learner: {base_learner}")


def get_final_model(base_learner, params):
    if base_learner in auto_knn_list():
        ridge_model = RidgeBoostedKNN(
            knn_params={
                **params,
                "base_learner": (lambda: RobustFaissKNNRegressor())(),
            }
        )
        return ridge_model
    if base_learner in ["OptimalLinearRankKNN"]:
        ridge_model = OptimalKNN(base_learner=RobustFaissKNNRegressor())
        return ridge_model
    raise ValueError(f"Unknown base learner: {base_learner}")


def auto_knn_list():
    return [
        "RidgeBoosted-AutoKNN",
        "RidgeBoosted-RankKNN",
        "RidgeBoosted-LinearRankKNN",
        "RidgeBoosted-PLSAutoKNN",
        "RidgeBoosted-LPPAutoKNN",
    ]
