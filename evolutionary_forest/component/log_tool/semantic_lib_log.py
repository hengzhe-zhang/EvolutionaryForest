from evolutionary_forest.multigene_gp import MultipleGeneGP


class SemanticLibLog:
    def __init__(self):
        self.semantic_lib_size_history = []
        self.semantic_lib_success_rate = []
        self.semantic_lib_pass = 0
        self.semantic_lib_forbid = 0
        self.best_individual_operator = []

    def success_rate_update(self):
        sum = self.semantic_lib_pass + self.semantic_lib_forbid
        if sum == 0:
            return
        self.semantic_lib_success_rate.append(self.semantic_lib_pass / sum)
        self.semantic_lib_pass = 0
        self.semantic_lib_forbid = 0

    def best_individual_update(self, hof):
        ind = hof[0]
        inf: MultipleGeneGP
        if not hasattr(ind, "generation_operator"):
            return
        if ind.generation_operator == "Evolution":
            self.best_individual_operator.append(1)
        else:
            self.best_individual_operator.append(0)
