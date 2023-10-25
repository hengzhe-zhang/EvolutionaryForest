class MultiFidelityEvaluation:
    def __init__(
        self, base_learner, n_gen, ps_tree_ratio, interleaving_period, verbose, **kwargs
    ):
        self.base_learner: str = base_learner
        self.n_gen: int = n_gen
        self.ps_tree_ratio: float = ps_tree_ratio
        self.interleaving_period: int = interleaving_period
        self.verbose = verbose

    def update_evaluation_mode(self, current_gen):
        # Full evaluation after some iterations
        if (
            self.interleaving_period is None or self.interleaving_period == 0
        ) and current_gen > self.n_gen * (1 - self.ps_tree_ratio):
            if isinstance(self.base_learner, str) and self.base_learner.startswith(
                "Fast-"
            ):
                self.base_learner = self.base_learner.replace("Fast-", "")

        if self.interleaving_period > 0:
            # Interleaving changing
            if current_gen % self.interleaving_period == 0:
                if not self.base_learner.startswith("Fast"):
                    self.base_learner = "Fast-" + self.base_learner
            else:
                self.base_learner = self.base_learner.replace("Fast-", "")
            if self.verbose:
                print(current_gen, self.base_learner, self.interleaving_period)
