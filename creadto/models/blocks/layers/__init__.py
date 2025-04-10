from typing import Dict, Union
import numpy as np
import torch
import torch.nn as nn


CONV_DIM_DICT = {
    1: nn.Conv1d,
    2: nn.Conv2d,
    3: nn.Conv3d
}
TRANSPOSE_CONV_DIM_DICT = {
    1: nn.ConvTranspose1d,
    2: nn.ConvTranspose2d,
    3: nn.ConvTranspose3d
}
BN_DIM_DICT = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
    3: nn.BatchNorm3d,
}


def build_activation(
    activ_cfg
) -> Union[nn.ReLU, nn.LeakyReLU, nn.PReLU]:
    ''' Builds activation functions
    '''
    if len(activ_cfg) == 0:
        return None
    activ_type = activ_cfg.get('type', 'relu')
    inplace = activ_cfg.get('inplace', False)
    if activ_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif activ_type == 'leaky-relu':
        leaky_relu_cfg = activ_cfg.get('leaky_relu', {})
        return nn.LeakyReLU(inplace=inplace, **leaky_relu_cfg)
    elif activ_type == 'prelu':
        prelu_cfg = activ_cfg.get('prelu', {})
        return nn.PReLU(inplace=inplace, **prelu_cfg)
    elif activ_type == 'none':
        return None
    else:
        raise ValueError(f'Unknown activation type: {activ_type}')


def build_norm_layer(
    input_dim: int,
    norm_cfg: Dict,
    dim: int = 1
) -> nn.Module:
    ''' Builds normalization modules
    '''
    if len(norm_cfg) == 0:
        return None
    norm_type = norm_cfg.get('type', 'bn')
    if norm_type == 'bn' or norm_type == 'batch-norm':
        bn_cfg = norm_cfg.get('batch_norm', {})
        if dim in BN_DIM_DICT:
            return BN_DIM_DICT[dim](input_dim, **bn_cfg)
        else:
            raise ValueError(f'Wrong dimension for BN: {dim}')
    elif norm_type == 'ln' or norm_type == 'layer-norm':
        layer_norm_cfg = norm_cfg.get('layer_norm', {})
        return nn.LayerNorm(input_dim, **layer_norm_cfg)
    elif norm_type == 'gn':
        group_norm_cfg = norm_cfg.get('group_norm', {})
        return nn.GroupNorm(num_channels=input_dim, **group_norm_cfg)
    elif norm_type.lower() == 'none':
        return None
    else:
        raise ValueError(f'Unknown normalization type: {norm_type}')


def build_rnn_cell(
    input_size: int,
    rnn_type='lstm',
    hidden_size=1024,
    bias=True
) -> Union[nn.LSTMCell, nn.GRUCell]:
    if rnn_type == 'lstm':
        return nn.LSTMCell(input_size, hidden_size=hidden_size, bias=bias)
    elif rnn_type == 'gru':
        return nn.GRUCell(input_size, hidden_size=hidden_size, bias=bias)
    else:
        raise ValueError(f'Unknown RNN type: {rnn_type}')

def identity(x):
    return x


def leaky_relu_0_2(x):
    return torch.nn.functional.leaky_relu(x, 0.2)


@torch.jit.script
def clamp_gain(x: torch.Tensor, g: float, c: float):
    return torch.clamp(x * g, -c, c)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulated_conv2d(x, weight, styles, padding=0, demodulate=True):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Calculate per-sample weights and demodulation coefficients.
    w = weight.unsqueeze(0)  # [NOIkk]
    w = w * styles.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute as one fused op using grouped convolution.
    batch_size = int(batch_size)
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = torch.nn.functional.conv2d(x, w, padding=padding, groups=batch_size)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    return x


activation_funcs = {
    "linear": {
        "fn": identity,
        "def_gain": 1
    },
    "lrelu": {
        "fn": leaky_relu_0_2,
        "def_gain": np.sqrt(2)
    }
}