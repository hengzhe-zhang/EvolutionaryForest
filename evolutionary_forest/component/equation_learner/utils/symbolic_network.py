"""Contains the symbolic regression neural network architecture."""
import numpy as np
import torch
import torch.nn as nn
from . import functions as functions


class SymbolicLayer(nn.Module):
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""

    def __init__(self, funcs=None, initial_weight=None, init_stddev=0.1, in_dim=None):
        """
        funcs: List of activation functions, using utils.functions
        initial_weight: (Optional) Initial value for weight matrix
        variable: Boolean of whether initial_weight is a variable or not
        init_stddev: (Optional) if initial_weight isn't passed in, this is standard deviation of initial weight
        """
        super().__init__()

        if funcs is None:
            funcs = functions.default_func
        self.initial_weight = initial_weight
        self.W = None  # Weight matrix
        self.built = False  # Boolean whether weights have been initialized

        self.output = None  # tensor for layer output
        self.n_funcs = len(funcs)  # Number of activation functions (and number of layer outputs)
        self.funcs = [func.torch for func in funcs]  # Convert functions to list of PyTorch functions
        self.n_double = functions.count_double(funcs)  # Number of activation functions that take 2 inputs
        self.n_single = self.n_funcs - self.n_double  # Number of activation functions that take 1 input

        self.out_dim = self.n_funcs + self.n_double

        if self.initial_weight is not None:  # use the given initial weight
            self.W = nn.Parameter(self.initial_weight.clone().detach())  # copies
            self.built = True
        else:
            self.W = torch.normal(mean=0.0, std=init_stddev, size=(in_dim, self.out_dim))

    def forward(self, x):  # used to be __call__
        """Multiply by weight matrix and apply activation units"""

        g = torch.matmul(x, self.W)  # shape = (?, self.size)
        self.output = []

        in_i = 0  # input index
        out_i = 0  # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            self.output.append(self.funcs[out_i](g[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            self.output.append(self.funcs[out_i](g[:, in_i], g[:, in_i + 1]))
            in_i += 2
            out_i += 1

        self.output = torch.stack(self.output, dim=1)

        return self.output

    def get_weight(self):
        return self.W.cpu().detach().numpy()

    def get_weight_tensor(self):
        return self.W.clone()


class SymbolicNet(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""

    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, init_stddev=0.1):
        super(SymbolicNet, self).__init__()

        self.depth = symbolic_depth  # Number of hidden layers
        self.funcs = funcs
        layer_in_dim = [1] + self.depth * [len(funcs)]

        if initial_weights is not None:
            layers = [SymbolicLayer(funcs=funcs, initial_weight=initial_weights[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())

        else:
            # Each layer initializes its own weights
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayer(funcs=self.funcs, init_stddev=init_stddev[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]

            # Initialize weights for last layer (without activation functions)
            self.output_weight = nn.Parameter(torch.rand((layers[-1].n_funcs, 1)))
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input):
        h = self.hidden_layers(input)  # Building hidden layers
        return torch.matmul(h, self.output_weight)  # Final output (no activation units) of network

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)] + \
            [self.output_weight.cpu().detach().numpy()]

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [self.hidden_layers[i].get_weight_tensor() for i in range(self.depth)] + \
            [self.output_weight.clone()]


class SymbolicLayerL0(SymbolicLayer):
    def __init__(self, in_dim=None, funcs=None, initial_weight=None, init_stddev=0.1,
                 bias=False, droprate_init=0.5, lamba=1.,
                 beta=2 / 3, gamma=-0.1, zeta=1.1, epsilon=1e-6):
        super().__init__(in_dim=in_dim, funcs=funcs, initial_weight=initial_weight, init_stddev=init_stddev)

        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.in_dim = in_dim
        self.eps = None

        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.epsilon = epsilon

        if self.use_bias:
            self.bias = nn.Parameter(0.1 * torch.ones((1, self.out_dim)))
        self.qz_log_alpha = nn.Parameter(torch.normal(mean=np.log(1 - self.droprate_init) - np.log(self.droprate_init),
                                                      std=1e-2, size=(in_dim, self.out_dim)))

    def quantile_concrete(self, u):
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(u) - torch.log(1.0 - u) + self.qz_log_alpha) / self.beta)
        return y * (self.zeta - self.gamma) + self.gamma

    def sample_u(self, shape, reuse_u=False):
        """Uniform random numbers for concrete distribution"""
        if self.eps is None or not reuse_u:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.eps = torch.rand(size=shape).to(device) * (1 - 2 * self.epsilon) + self.epsilon
        return self.eps

    def sample_z(self, batch_size, sample=True):
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return torch.clamp(z, min=0, max=1)
        else:  # Mean of the hard concrete distribution
            pi = torch.sigmoid(self.qz_log_alpha)
            return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def get_z_mean(self):
        """Mean of the hard concrete distribution"""
        pi = torch.sigmoid(self.qz_log_alpha)
        return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def sample_weights(self, reuse_u=False):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim), reuse_u=reuse_u))
        mask = torch.clamp(z, min=0.0, max=1.0)
        return mask * self.W

    def get_weight(self):
        """Deterministic value of weight based on mean of z"""
        return self.W * self.get_z_mean()

    def loss(self):
        """Regularization loss term"""
        return torch.sum(torch.sigmoid(self.qz_log_alpha - self.beta * np.log(-self.gamma / self.zeta)))

    def forward(self, x, sample=True, reuse_u=False):
        """Multiply by weight matrix and apply activation units"""
        if sample:
            h = torch.matmul(x, self.sample_weights(reuse_u=reuse_u))
        else:
            w = self.get_weight()
            h = torch.matmul(x, w)

        if self.use_bias:
            h = h + self.bias

        # shape of h = (?, self.n_funcs)

        output = []
        # apply a different activation unit to each column of h
        in_i = 0  # input index
        out_i = 0  # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            output.append(self.funcs[out_i](h[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            output.append(self.funcs[out_i](h[:, in_i], h[:, in_i + 1]))
            in_i += 2
            out_i += 1
        output = torch.stack(output, dim=1)
        return output


class SymbolicNetL0(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""

    def __init__(self, symbolic_depth, in_dim=1, funcs=None, initial_weights=None, init_stddev=0.1):
        super(SymbolicNetL0, self).__init__()
        self.depth = symbolic_depth  # Number of hidden layers
        self.funcs = funcs

        layer_in_dim = [in_dim] + self.depth * [len(funcs)]
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=funcs, initial_weight=initial_weights[i],
                                      in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())
        else:
            # Each layer initializes its own weights
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayerL0(funcs=funcs, init_stddev=init_stddev[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            # Initialize weights for last layer (without activation functions)
            self.output_weight = nn.Parameter(torch.rand(size=(self.hidden_layers[-1].n_funcs, 1)) * 2)
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input, sample=True, reuse_u=False):
        # connect output from previous layer to input of next layer
        h = input
        for i in range(self.depth):
            h = self.hidden_layers[i](h, sample=sample, reuse_u=reuse_u)

        h = torch.matmul(h, self.output_weight)  # Final output (no activation units) of network
        return h

    def get_loss(self):
        return torch.sum(torch.stack([self.hidden_layers[i].loss() for i in range(self.depth)]))

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [self.hidden_layers[i].get_weight().cpu().detach().numpy() for i in range(self.depth)] + \
            [self.output_weight.cpu().detach().numpy()]
