import numpy as np

from evolutionary_forest.application.time_series.tiny_step_ts import (
    AugmentedBaselineIncrementalForecaster,
)
from evolutionary_forest.learners import RidgeEvolutionaryFeatureLearner

if __name__ == "__main__":
    time = np.linspace(0, 10, 50)
    amplitude = np.sin(2 * np.pi * time)
    decomposed_forecaster = AugmentedBaselineIncrementalForecaster(
        lag=4,
        model=RidgeEvolutionaryFeatureLearner(),
        verbose=True,
        augment_interval=1,
    )

    decomposed_forecaster.fit(amplitude)
    print(decomposed_forecaster.predict(amplitude, 5))
