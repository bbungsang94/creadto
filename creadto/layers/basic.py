import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as nninit
from creadto.layers import activation_funcs, build_activation, build_norm_layer


def init_weights(layer, name='', init_type='xavier', distr='uniform', gain=1.0, activ_type='leaky-relu', lrelu_slope=0.01, **kwargs):
    if len(name) < 1:
        name = str(layer)
    weights = layer.weight
    if init_type == 'xavier':
        if distr == 'uniform':
            nninit.xavier_uniform_(weights, gain=gain)
        elif distr == 'normal':
            nninit.xavier_normal_(weights, gain=gain)
        else:
            raise ValueError(
                'Unknown distribution "{}" for Xavier init'.format(distr))
    elif init_type == 'kaiming':
        activ_type = activ_type.replace('-', '_')
        if distr == 'uniform':
            nninit.kaiming_uniform_(weights, a=lrelu_slope,
                                    nonlinearity=activ_type)
        elif distr == 'normal':
            nninit.kaiming_normal_(weights, a=lrelu_slope,
                                   nonlinearity=activ_type)
        else:
            raise ValueError(
                'Unknown distribution "{}" for Kaiming init'.format(distr))

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


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers = [1024, 1024], activation = None,
                 normalization = None, dropout: float = 0.5, gain: float = 0.01, preactivated: bool = False,
                 flatten: bool = True, **kwargs) -> None:
        ''' Simple MLP layer
        '''
        super(MLP, self).__init__()
        if layers is None:
            layers = []
        self.flatten = flatten
        self.input_dim = input_dim
        self.output_dim = output_dim

        if activation is None:
            activation = {}
        if normalization is None:
            normalization = {}

        curr_input_dim = input_dim
        self.num_layers = len(layers)
        self.blocks = []
        for layer_idx, layer_dim in enumerate(layers):
            activ = build_activation(activation)
            norm_layer = build_norm_layer(
                layer_dim, norm_cfg=normalization, dim=1)
            bias = norm_layer is None

            linear = nn.Linear(curr_input_dim, layer_dim, bias=bias)
            curr_input_dim = layer_dim

            layer = []
            if preactivated:
                if norm_layer is not None:
                    layer.append(norm_layer)

                if activ is not None:
                    layer.append(activ)

                layer.append(linear)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))
            else:
                layer.append(linear)

                if activ is not None:
                    layer.append(activ)

                if norm_layer is not None:
                    layer.append(norm_layer)

                if dropout > 0.0:
                    layer.append(nn.Dropout(dropout))

            block = nn.Sequential(*layer)
            self.add_module('layer_{:03d}'.format(layer_idx), block)
            self.blocks.append(block)

        self.output_layer = nn.Linear(curr_input_dim, output_dim)
        init_weights(
            self.output_layer, gain=gain,
            activ_type=activation.get('type', 'none'),
            init_type='xavier', distr='uniform')

    def extra_repr(self):
        msg = [
            f'Input ({self.input_dim}) -> Output ({self.output_dim})',
            f'Flatten: {self.flatten}',
        ]

        return '\n'.join(msg)

    def forward(self, module_input: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = module_input.shape[0]
        # Flatten all dimensions
        curr_input = module_input
        if self.flatten:
            curr_input = curr_input.view(batch_size, -1)
        for block in self.blocks:
            curr_input = block(curr_input)
        return self.output_layer(curr_input)