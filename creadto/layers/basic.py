import numpy as np
import torch
import torch.nn as nn

from creadto.layers import activation_funcs


class FullyConnectedLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation='linear', lr_multiplier=1, bias_init=0):
        super().__init__()

        self.activation = activation_funcs[activation]['fn']
        self.activation_gain = activation_funcs[activation]['def_gain']
        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1:
            b = b * self.bias_gain
        x = self.activation(torch.addmm(b.unsqueeze(0), x, w.t())) * self.activation_gain
        return x