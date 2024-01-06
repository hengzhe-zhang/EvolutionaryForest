from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from evolutionary_forest.model.sample import ddpm_sample
from evolutionary_forest.model.train import ddpm_train

model_params = {
    "num_classes": 0,
    "is_y_cond": True,
    "rtdl_params": {
        "d_layers": [
            256,
            1024,
            1024,
            1024,
            1024,
            512,
        ],
        "dropout": 0.0,
    },
}


class DDPM:
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.model = ddpm_train(
            X,
            y,
            T_dict={"normalization": "minmax"},
            model_params=model_params,
        )

    def sample(self):
        ddpm_sample(
            self.X,
            self.y,
            self.model,
            T_dict={"normalization": "minmax"},
            model_params=model_params,
        )


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=100, random_state=0
    )
    ddpm = DDPM()
    ddpm.fit(X, y)
    print(ddpm.sample())
