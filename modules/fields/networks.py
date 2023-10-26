import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_activation(activation):
    if activation == 'identity':
        return lambda x: x
    elif activation == 'relu':
        return lambda x: F.relu(x)
    else:
        raise NotImplementedError


class VanillaMLP(nn.Module):
    def __init__(self,
                 dim_in,
                 dim_out,
                 n_neurons,
                 n_hidden_layers,
                 sphere_init=False,
                 weight_norm=False,
                 sphere_init_radius=0.5,
                 output_activation='identity'):
        super().__init__()
        self.n_neurons, self.n_hidden_layers = n_neurons, n_hidden_layers
        self.sphere_init, self.weight_norm = sphere_init, weight_norm
        self.sphere_init_radius = sphere_init_radius
        self.layers = [self.make_linear(dim_in, self.n_neurons, is_first=True, is_last=False), self.make_activation()]
        for i in range(self.n_hidden_layers - 1):
            self.layers += [self.make_linear(self.n_neurons, self.n_neurons, is_first=False, is_last=False),
                            self.make_activation()]
        self.layers += [self.make_linear(self.n_neurons, dim_out, is_first=False, is_last=True)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x.float())
        return -x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)  # network without bias will degrade quality
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(dim_in), std=0.0001)
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out))
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)
