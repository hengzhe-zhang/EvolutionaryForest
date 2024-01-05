from ctgan.synthesizers.ctgan import *
from geomloss import SamplesLoss
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.nn.functional import nll_loss, mse_loss
from tqdm import tqdm

from evolutionary_forest.utility.wasserstein_distance import (
    covariance_matrix,
    wasserstein_distance_torch,
)


class IndexDataSampler(DataSampler):
    def sample_data(self, n, col, opt, return_index=False):
        """Sample data from original training data satisfying the sampled conditional vector.

        Returns:
            n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(self._data), size=n)
            if return_index:
                return self._data[idx], idx
            else:
                return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))
        if return_index:
            return self._data[idx], idx
        else:
            return self._data[idx]


class SyncDataTransformer(DataTransformer):
    def transform(self, raw_data):
        """Take raw data and output a matrix data."""
        if not isinstance(raw_data, pd.DataFrame):
            column_names = [str(num) for num in range(raw_data.shape[1])]
            raw_data = pd.DataFrame(raw_data, columns=column_names)

        # Only use parallelization with larger data sizes.
        # Otherwise, the transformation will be slower.
        if raw_data.shape[0] < 5000:
            column_data_list = self._synchronous_transform(
                raw_data, self._column_transform_info_list
            )
        else:
            column_data_list = self._parallel_transform(
                raw_data, self._column_transform_info_list
            )

        return np.concatenate(column_data_list, axis=1).astype(float)


class ASGAN(CTGAN):
    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        learn_from_teacher=None,
        assisted_loss=None,
        weight_of_distance: float = 1,
    ):
        super().__init__(
            embedding_dim,
            generator_dim,
            discriminator_dim,
            generator_lr,
            generator_decay,
            discriminator_lr,
            discriminator_decay,
            batch_size,
            discriminator_steps,
            log_frequency,
            verbose,
            epochs,
            pac,
            cuda,
        )
        self.learn_from_teacher = learn_from_teacher
        self.assisted_loss = assisted_loss
        self.weight_of_distance = weight_of_distance

    # def fit(self, train_data, train_label, discrete_columns=(), epochs=None):
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        # Train a model
        forest = ExtraTreesRegressor()
        train_data, train_label = train_data, train_data[:, -1]
        train_label = (
            StandardScaler().fit_transform(train_label.reshape(-1, 1)).reshape(-1)
        )
        forest.fit(train_data[:, :-1], train_label)

        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        self._transformer = SyncDataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)

        self._data_sampler = IndexDataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
        )

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim,
        ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac,
        ).to(self._device)

        # use the same architecture as discriminator
        int_dimension = self._transformer.output_info_list[-1][0].dim
        cluster_dimension = self._transformer.output_info_list[-1][1].dim
        label_dimension = cluster_dimension + int_dimension
        learner = Discriminator(
            data_dim - label_dimension + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=1,
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        optimizerL = optim.Adam(
            learner.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(
            columns=["Epoch", "Generator Loss", "Distriminator Loss"]
        )

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = "Gen. ({gen:.2f}) | Discrim. ({dis:.2f}) "
            epoch_iterator.set_description(description.format(gen=0, dis=0, lea=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real, real_index = self._data_sampler.sample_data(
                            self._batch_size, col, opt, return_index=True
                        )
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real, real_index = self._data_sampler.sample_data(
                            self._batch_size, col[perm], opt[perm], return_index=True
                        )
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype("float32")).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward(retain_graph=True)
                    optimizerD.step()

                    if self.learn_from_teacher is not None:
                        if self.learn_from_teacher == "Real":
                            real_pred = learner(real_cat[:, :-label_dimension])
                            real_label = torch.from_numpy(
                                train_label[real_index].astype("float32")
                            ).to(self._device)
                            # MSE
                            loss_l = mse_loss(real_pred.view(-1), real_label)
                        elif self.learn_from_teacher == "Fake":
                            fake_pred = learner(fake_cat[:, :-label_dimension])
                            _, fake_target = self.get_prediction(fake_cat, forest)
                            # MSE
                            loss_l = mse_loss(fake_pred.view(-1), fake_target)
                        else:
                            raise Exception

                        optimizerL.zero_grad(set_to_none=False)
                        loss_l.backward()
                        optimizerL.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                # fake_pred = learner(fakeact)
                fake_data, fake_target = self.get_prediction(fakeact, forest)
                generated_data = np.concatenate(
                    (fake_data, fake_target.reshape(-1, 1)), axis=1
                )
                generated_data = self._transformer.transform(generated_data).astype(
                    np.float32
                )
                if self.learn_from_teacher is None:
                    # minimize learner loss to generate real samples
                    learner_loss = mse_loss(
                        fakeact[:, -(cluster_dimension + int_dimension)],
                        torch.from_numpy(
                            generated_data[:, -(cluster_dimension + int_dimension)]
                        ).to(self._device),
                    ) + nll_loss(
                        fakeact[:, -cluster_dimension:],
                        torch.from_numpy(
                            np.argmax(generated_data[:, -cluster_dimension:], axis=1)
                        ).to(self._device),
                    )
                else:
                    learner_loss = mse_loss(
                        learner(fakeact[:, :-label_dimension]).view(-1), fake_target
                    )

                train_data_torch = torch.from_numpy(train_data.astype("float32")).to(
                    self._device
                )
                distance_loss = self.feature_matching_loss(
                    fakeact, train_data, train_data_torch
                )
                scaling_weight = (
                    abs(torch.mean(y_fake).detach().cpu())
                ) / distance_loss.detach().cpu()
                print("Scaling Weight", scaling_weight)
                loss_g = (
                    -torch.mean(y_fake)
                    + cross_entropy
                    # + self.weight_of_distance * learner_loss
                    + self.weight_of_distance * scaling_weight * distance_loss
                )

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

            generator_loss = loss_g.detach().cpu()
            discriminator_loss = loss_d.detach().cpu()

            epoch_loss_df = pd.DataFrame(
                {
                    "Epoch": [i],
                    "Generator Loss": [generator_loss],
                    "Discriminator Loss": [discriminator_loss],
                }
            )
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(
                        gen=generator_loss,
                        dis=discriminator_loss,
                    )
                )

    def get_prediction(self, fake_cat, forest):
        fake_data = self._transformer.inverse_transform(fake_cat.detach().numpy())
        fake_target = torch.from_numpy(
            forest.predict(fake_data[:, :-1]).astype("float32")
        ).to(self._device)
        return fake_data, fake_target

    def feature_matching_loss(self, fakeact, train_data, train_data_torch):
        index = 0
        coarse_index = []
        for info in self._transformer.output_info_list:
            index += info[0].dim
            coarse_index.extend([i for i in range(index, index + info[1].dim)])
            index += info[1].dim
        coarse_index = torch.tensor(coarse_index)
        if self.assisted_loss == "Mean":
            distance_loss = torch.mean(
                (
                    torch.mean(fakeact[:, coarse_index], dim=0)
                    - torch.mean(train_data_torch[:, coarse_index], dim=0)
                )
                ** 2
            )
            # print("Distance Loss", distance_loss)
        elif self.assisted_loss == "SD":
            loss = SamplesLoss(loss="sinkhorn")
            distance_loss = loss(
                fakeact[:, coarse_index], train_data_torch[:, coarse_index]
            )
        elif self.assisted_loss == "HD":
            loss = SamplesLoss(loss="hausdorff")
            distance_loss = loss(
                fakeact[:, coarse_index], train_data_torch[:, coarse_index]
            )
        elif self.assisted_loss == "EMMD":
            loss = SamplesLoss(loss="energy")
            distance_loss = loss(
                fakeact[:, coarse_index], train_data_torch[:, coarse_index]
            )
        elif self.assisted_loss == "GMMD":
            loss = SamplesLoss(loss="gaussian")
            distance_loss = loss(
                fakeact[:, coarse_index], train_data_torch[:, coarse_index]
            )
        elif self.assisted_loss == "LMMD":
            loss = SamplesLoss(loss="laplacian")
            distance_loss = loss(
                fakeact[:, coarse_index], train_data_torch[:, coarse_index]
            )
        elif self.assisted_loss == "WA":
            mean_fake = torch.mean(fakeact, dim=0)
            mean_train = torch.mean(train_data_torch, dim=0)
            cov_fake = covariance_matrix(fakeact, mean_fake)
            cov_train = covariance_matrix(train_data_torch, mean_train)
            distance_loss = wasserstein_distance_torch(
                mean_fake, mean_train, cov_fake, cov_train
            )
        elif self.assisted_loss == "KL":
            std1 = torch.std(train_data_torch, dim=0)
            std2 = torch.std(fakeact, dim=0)
            mean1 = torch.mean(train_data_torch, dim=0)
            mean2 = torch.mean(fakeact, dim=0)
            eps = torch.finfo(torch.float32).eps
            std1 = std1 + eps
            std2 = std2 + eps
            log_term = torch.log(std2 / std1)
            variance_term = (std1**2 + (mean1 - mean2) ** 2) / (2 * std2**2)
            distance_loss = torch.mean(log_term + variance_term)
        else:
            distance_loss = 0
        return distance_loss


if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    ctgan = ASGAN(
        epochs=100, verbose=True, assisted_loss="Mean", weight_of_distance=0.1
    )
    ctgan.fit(X)
    print(ctgan.sample(50))
