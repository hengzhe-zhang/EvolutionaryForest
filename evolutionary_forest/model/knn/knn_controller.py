from evolutionary_forest.model.knn.FaissKNNRegressor import (
    FaissKNNLinearRankRegressor,
    FaissKNNRankRegressor,
    RobustFaissKNNRegressor,
    AdaptiveRNRegressor,
)
from evolutionary_forest.model.optimal_knn.GBOptimalKNN import RidgeBoostedKNN


def adaptive_neighbors(base_learner, params, X, y):
    if (
        base_learner == "RidgeBoosted-LinearRankKNN"
        and params["n_neighbors"] == "Adaptive"
    ):
        # only need to check once
        r = RobustFaissKNNRegressor()
        r.fit(X, y)
        if r.k_opt_ < 5:
            params["n_neighbors"] = 10
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
    raise ValueError(f"Unknown base learner: {base_learner}")


def get_final_model(base_learner, params):
    if base_learner in [
        "RidgeBoosted-SafeKNN",
        "RidgeBoosted-RankKNN",
        "RidgeBoosted-LinearRankKNN",
    ]:
        ridge_model = RidgeBoostedKNN(
            knn_params={
                **params,
                "base_learner": (lambda: RobustFaissKNNRegressor())(),
            }
        )
        return ridge_model
    raise ValueError(f"Unknown base learner: {base_learner}")
