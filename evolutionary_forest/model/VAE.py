import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import TransformerMixin
from sklearn.datasets import make_classification
from skorch import NeuralNet
from skorch.callbacks import EarlyStopping
from skorch.utils import to_numpy
from torch import optim
from torch.nn import MSELoss


class VAE(nn.Module):
    def __init__(self, input_unit, hidden_layer=64, embedding_size=2):
        super(VAE, self).__init__()
        self.input_unit = input_unit
        self.hidden_layer = hidden_layer
        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(input_unit, hidden_layer)
        self.fc21 = nn.Linear(hidden_layer, embedding_size)
        self.fc22 = nn.Linear(hidden_layer, embedding_size)
        self.fc3 = nn.Linear(embedding_size, hidden_layer)
        self.fc4 = nn.Linear(hidden_layer, input_unit)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3).view(-1, self.input_unit)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_unit))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class NeuralNetTransformer(NeuralNet, TransformerMixin):
    def get_loss(self, y_pred, y_true, X, **kwargs):
        recons, mu, log_var = y_pred
        self.beta = 0.1
        recons_loss = F.mse_loss(recons, X)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + self.beta * kld_loss
        return loss

    def transform(self, X):
        X = X.astype(np.float32)
        out = []
        for outs in self.forward_iter(X, training=False):
            outs = outs[1] if isinstance(outs, tuple) else outs
            out.append(to_numpy(outs))
        transforms = np.concatenate(out, 0)
        return transforms


if __name__ == '__main__':
    X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X, y = X.astype(np.float32), y.astype(np.int64)

    vae = NeuralNetTransformer(VAE,
                               criterion=MSELoss(),
                               optimizer=optim.Adam,
                               module__input_unit=X.shape[1],
                               max_epochs=1000,
                               callbacks=[EarlyStopping(patience=20)],
                               verbose=True)
    vae.fit(X, y)
    print(vae.transform(X))
