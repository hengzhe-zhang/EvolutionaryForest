from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import wilcoxon

from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utility.statistics.feature_count import number_of_used_features

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


def log_check(ef: "EvolutionaryForestRegressor"):
    log_item = ef.log_item
    if "NumberOfFeatureInBestModel" in log_item:
        ef.best_individual_number_of_features.append(
            number_of_used_features(ef.hof, ef.pset)
        )

    if log_item != "":
        pop = ef.pop
        best_ind: MultipleGeneGP = sorted(pop, key=lambda x: x.fitness.wvalues[0])[-1]

        if "LOOCV" in log_item:
            # leave-one-out cross-validation loss
            ef.loocv_logs.append(best_ind.fitness.wvalues[0])

        if "SharpnessRatio" in log_item:
            # ratio between sharpness and SAM loss
            ef.sharpness_ratio_logs.append(
                best_ind.fitness.values[1] / best_ind.sam_loss
            )

        if "SAM" in log_item:
            # sharpness value
            ef.sharpness_logs.append(best_ind.fitness.values[1])

        if "Duel" in log_item:
            # duel p-value
            if not np.all(np.equal(best_ind.case_values, ef.hof[0].case_values)):
                p_value = wilcoxon(
                    best_ind.case_values / ef.hof[0].case_values,
                    np.ones(len(best_ind.case_values)),
                    alternative="less",
                )[1]
                ef.duel_logs.append(p_value)
            else:
                ef.duel_logs.append(0)
