# A simple example for saving the final model
import dill
from evolutionary_forest.forest import EvolutionaryForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
ef = EvolutionaryForestRegressor(
    max_height=3,
    normalize=True,
    select="AutomaticLexicase",
    gene_num=10,
    boost_size=100,
    n_gen=2,
    n_pop=10,
    cross_pb=1,
    base_learner="Random-DT",
    verbose=True,
)
ef.fit(x_train, y_train)
print(r2_score(y_test, ef.predict(x_test)))

with open("model.pkl", "wb") as f:
    if hasattr(ef, "pool"):
        del ef.pool
    dill.dump(ef, f)

with open("model.pkl", "rb") as f:
    ef = dill.load(f)

print(r2_score(y_test, ef.predict(x_test)))
