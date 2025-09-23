from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evolutionary_forest.forest import EvolutionaryForestRegressor


class MultiFidelityEvaluation:
    def __init__(
        self,
        algorithm: "EvolutionaryForestRegressor",
        n_gen,
        interleaving_period,
        verbose,
        sr_tree_ratio=0,
        **kwargs,
    ):
        self.algorithm = algorithm
        self.n_gen: int = n_gen
        self.sr_tree_ratio: float = sr_tree_ratio
        self.interleaving_period: int = interleaving_period
        self.verbose = verbose
        self.legacy_parameters(kwargs)

    def update_evaluation_mode(self, current_gen):
        self.fast_evaluation_after_several_generations(current_gen)
        self.interleave_fast_evaluation(current_gen)

    def fast_evaluation_after_several_generations(self, current_gen):
        # Full evaluation after some iterations
        if (
            self.interleaving_period is None or self.interleaving_period == 0
        ) and current_gen > self.n_gen * (1 - self.sr_tree_ratio):
            if isinstance(
                self.algorithm.base_learner, str
            ) and self.algorithm.base_learner.startswith("Fast-"):
                self.algorithm.base_learner = self.algorithm.base_learner.replace(
                    "Fast-", ""
                )

    def interleave_fast_evaluation(self, current_gen):
        if self.interleaving_period > 0:
            # Interleaving changing
            if current_gen % self.interleaving_period == 0:
                if not self.algorithm.base_learner.startswith("Fast"):
                    self.algorithm.base_learner = "Fast-" + self.algorithm.base_learner
            else:
                self.algorithm.base_learner = self.algorithm.base_learner.replace(
                    "Fast-", ""
                )
            if self.verbose:
                print(
                    current_gen, self.algorithm.base_learner, self.interleaving_period
                )

    def legacy_parameters(self, kwargs):
        # some legacy parameters
        self.sr_tree_ratio = kwargs.get("ps_tree_ratio", self.sr_tree_ratio)
