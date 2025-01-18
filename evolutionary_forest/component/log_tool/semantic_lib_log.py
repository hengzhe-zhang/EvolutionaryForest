from evolutionary_forest.multigene_gp import MultipleGeneGP
from evolutionary_forest.utility.tree_pool import SemanticLibrary


class SemanticLibLog:
    def __init__(self):
        self.semantic_lib_size_history = []
        self.semantic_lib_success_rate = []
        self.semantic_lib_success = 0
        self.semantic_lib_fail = 0
        self.best_individual_operator = []
        self.unique_trees_in_pop = []
        self.unique_fitness_in_pop = []
        self.semantic_lib_usage = []
        self.best_tree_size_history = []

    def success_rate_update(self):
        sum = self.semantic_lib_success + self.semantic_lib_fail
        if sum == 0:
            return
        self.semantic_lib_success_rate.append(self.semantic_lib_success / sum)
        self.semantic_lib_success = 0
        self.semantic_lib_fail = 0

    def best_individual_update(self, hof):
        ind = hof[0]
        inf: MultipleGeneGP
        if not hasattr(ind, "generation_operator"):
            return
        if ind.generation_operator == "Evolution":
            self.best_individual_operator.append(1)
        else:
            self.best_individual_operator.append(0)

    def unique_trees_in_pop_update(self, pop):
        unique_trees = set()
        for ind in pop:
            ind: MultipleGeneGP
            for tree in ind.gene:
                unique_trees.add(str(tree))
        self.unique_trees_in_pop.append(len(unique_trees))

    def unique_fitness_in_pop_update(self, pop):
        unique_trees = set()
        for ind in pop:
            ind: MultipleGeneGP
            unique_trees.add(ind.fitness.wvalues)
        self.unique_fitness_in_pop.append(len(unique_trees))

    def semantic_lib_usage_update(self, lib: SemanticLibrary):
        used = [
            k for k, v in lib.frequency.items() if v > 0
        ]  # Count all keys with v > 0
        self.semantic_lib_usage.append(len(used))

    def update_best_tree_size(self, hof):
        assert len(hof) == 1, (
            "Hall of Fame (hof) should contain exactly one individual."
        )

        # Calculate and return the size of the tree
        best_tree = hof[0]
        size = sum([len(tree) for tree in best_tree.gene])

        self.best_tree_size_history.append(size)
