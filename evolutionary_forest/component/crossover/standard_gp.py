import numpy as np
import time
from deap import base, creator, gp, tools, algorithms
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import operator
import random


def protected_div(a, b):
    return np.where(b != 0, a / b, 1)


def protected_log(a):
    return np.where(np.abs(a) > 0, np.log(np.abs(a)), 0)


def protected_sqrt(a):
    return np.sqrt(np.abs(a))


class SimpleGP:
    def __init__(self, n_features):
        self.n_features = n_features
        self.pset = self._build_primitives()
        self.toolbox = self._setup_toolbox()

    def _build_primitives(self):
        pset = gp.PrimitiveSet("MAIN", arity=self.n_features)
        pset.addPrimitive(np.add, 2)
        pset.addPrimitive(np.subtract, 2)
        pset.addPrimitive(np.multiply, 2)
        pset.addPrimitive(protected_div, 2)
        pset.addPrimitive(np.negative, 1)
        pset.addPrimitive(protected_sqrt, 1)
        pset.addPrimitive(protected_log, 1)
        pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))
        for i in range(self.n_features):
            pset.renameArguments(**{f"ARG{i}": f"x{i}"})
        return pset

    def _setup_toolbox(self):
        if "FitnessMin" in creator.__dict__:
            del creator.FitnessMin
        if "Individual" in creator.__dict__:
            del creator.Individual

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=4)
        toolbox.register(
            "individual", tools.initIterate, creator.Individual, toolbox.expr
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.eval_regression)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=self.pset)
        toolbox.decorate(
            "mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        toolbox.decorate(
            "mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)
        )
        return toolbox

    def eval_regression(self, individual):
        # Compile the GP individual to a callable that produces a single feature
        func = gp.compile(individual, self.pset)
        try:
            # Construct GP feature for training set
            feature = func(*self.X_train.T)
            feature = np.broadcast_to(feature, (self.X_train.shape[0],))

            # Validate feature
            if not np.all(np.isfinite(feature)) or not np.all(np.isreal(feature)):
                return (1e6,)

            X_feat = feature.reshape(-1, 1)

            # Fit a Ridge regressor on the GP-constructed feature
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_feat, self.y_train)
            preds = ridge.predict(X_feat)

            # Use MSE as the fitness to minimize (keeps compatibility with existing setup)
            mse = mean_squared_error(self.y_train, preds)
            return (mse,)
        except Exception:
            # If evaluation fails for any individual, give it a large penalty
            return (1e6,)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def run_evolution(self, pop_size=100, n_gen=50):
        population = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        start = time.time()
        pop, log = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=0.9,
            mutpb=0.1,
            ngen=n_gen,
            stats=mstats,
            halloffame=hof,
            verbose=True,
        )
        elapsed = time.time() - start

        best = hof[0]
        return best, log, elapsed


def run_simple_gp(n_gen=50, pop_size=100):
    # Load data
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize GP
    gp_model = SimpleGP(n_features=X_train.shape[1])
    gp_model.fit(X_train, y_train)

    # Run evolution
    best_individual, log, time_taken = gp_model.run_evolution(
        pop_size=pop_size, n_gen=n_gen
    )

    # Evaluate best individual by chaining a Ridge on top of the GP-constructed feature
    func = gp.compile(best_individual, gp_model.pset)

    # Training feature and Ridge fit
    train_feature = func(*gp_model.X_train.T)
    train_feature = np.broadcast_to(train_feature, (gp_model.X_train.shape[0],))
    X_train_feat = train_feature.reshape(-1, 1)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_feat, y_train)
    train_preds = ridge.predict(X_train_feat)
    r2_train = r2_score(y_train, train_preds)

    # Test feature and predictions
    test_feature = func(*X_test.T)
    test_feature = np.broadcast_to(test_feature, (X_test.shape[0],))
    X_test_feat = test_feature.reshape(-1, 1)
    test_preds = ridge.predict(X_test_feat)
    r2_test = r2_score(y_test, test_preds)

    print(f"Best individual: {best_individual}")
    print(f"Train R²: {r2_train:.4f}")
    print(f"Test R²: {r2_test:.4f}")
    print(f"Time taken: {time_taken:.2f}s")

    return best_individual, r2_train, r2_test


if __name__ == "__main__":
    print("=" * 60)
    print("Simple Genetic Programming for Regression")
    print("=" * 60)
    run_simple_gp(n_gen=10, pop_size=100)
    print("=" * 60)
