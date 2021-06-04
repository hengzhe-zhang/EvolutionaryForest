import numpy as np
from deap.tools import HallOfFame


class LexicaseHOF(HallOfFame):
    def __init__(self):
        HallOfFame.__init__(self, None)

    def update(self, population):
        for p in population:
            self.insert(p)

        hofer_arr: np.ndarray = None
        for i, hofer in enumerate(self):
            if hofer_arr is None:
                hofer_arr = np.array(-1 * hofer.case_values).reshape(1, -1)
            else:
                hofer_arr = np.concatenate([hofer_arr, np.array(-1 * hofer.case_values).reshape(1, -1)])
        max_hof = np.max(hofer_arr, axis=0)

        del_ind = []
        max_value = np.full_like(max_hof, -np.inf)
        for index, x in enumerate(self):
            fitness_wvalues = np.array(-1 * x.case_values)
            if np.any(fitness_wvalues >= max_hof) \
                and np.any(fitness_wvalues > max_value):
                loc = np.where(fitness_wvalues > max_value)
                max_value[loc] = fitness_wvalues[loc]
                continue
            del_ind.append(index)

        for i in reversed(del_ind):
            self.remove(i)


class RandomObjectiveHOF(HallOfFame):
    """
    An archive which updating based on random projected fitness values
    """

    def update(self, population):
        for ind in population:
            if len(self) == 0 and self.maxsize != 0:
                # Working on an empty hall of fame is problematic for the
                # "for else"
                self.insert(population[0])
                continue
            if np.any(ind.case_values < self[-1].case_values) or len(self) < self.maxsize:
                for hofer in self:
                    # Loop through the hall of fame to check for any
                    # similar individual
                    if self.similar(ind, hofer):
                        break
                else:
                    # The individual is unique and strictly better than
                    # the worst
                    if len(self) >= self.maxsize:
                        self.remove(-1)
                    self.insert(ind)


class AutoLexicaseHOF(HallOfFame):
    """
    Automatically archive the best indivdiaul on each objective value
    """

    def update(self, population):
        """
        Update the Pareto front hall of fame with the *population* by adding
        the individuals from the population that are not dominated by the hall
        of fame. If any individual in the hall of fame is dominated it is
        removed.

        :param population: A list of individual with a fitness attribute to
                           update the hall of fame with.
        """
        # insert all individuals
        for p in population:
            self.insert(p)

        # calculate minimum value on each dimension
        hofer_arr: np.ndarray = None
        for i, hofer in enumerate(self):
            if hofer_arr is None:
                hofer_arr = np.array(hofer.case_values).reshape(1, -1)
            else:
                hofer_arr = np.concatenate([hofer_arr, np.array(hofer.case_values).reshape(1, -1)])
        max_hof = np.min(hofer_arr, axis=0)

        # only append individuals which can make a contribution
        del_ind = []
        max_value = np.full_like(max_hof, np.inf)
        for index, x in enumerate(self):
            fitness_wvalues = np.array(x.case_values)
            if np.any(fitness_wvalues <= max_hof) and np.any(fitness_wvalues < max_value):
                loc = np.where(fitness_wvalues < max_value)
                max_value[loc] = fitness_wvalues[loc]
                continue
            del_ind.append(index)

        for i in reversed(del_ind):
            self.remove(i)
