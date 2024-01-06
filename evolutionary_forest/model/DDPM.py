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
            256,
            # 1024,
            # 1024,
            # 1024,
            256,
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
            lr=0.01,
            steps=100,
            num_timesteps=100,
            T_dict={"normalization": "minmax"},
            model_params=model_params,
        )

    def sample(self):
        return ddpm_sample(
            self.X,
            self.y,
            self.model,
            num_samples=10,
            num_timesteps=100,
            num_numerical_features=self.X.shape[1],
            T_dict={"normalization": "minmax"},
            model_params=model_params,
        )


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    # X = StandardScaler().fit_transform(X)
    # y = StandardScaler().fit_transform(y.reshape(-1, 1)).reshape(-1)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=100, random_state=0
    )
    ddpm = DDPM()
    ddpm.fit(X, y)
    print(ddpm.sample())
